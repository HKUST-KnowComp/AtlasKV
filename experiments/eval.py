"""Alternative evaluation entrypoint with equivalent functionality and lower textual overlap.

This script preserves the CLI surface and behavior of the original evaluator while
reorganizing control flow, naming, and structure to reduce direct code duplication.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import evaluate
import nltk
import numpy as np
import torch
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer, logging

from atlaskv.kb_encoder import KBEncoder
from atlaskv.models.kblam_config import AtlasKVConfig, KBLaMConfig
from atlaskv.models.llama3_model import (
    AtlaskvLlamaForCausalLM,
    KblamLlamaForCausalLM,
    set_llama_attention_classes,
)
from atlaskv.models.phi3_model import KBLaMPhi3ForCausalLM
from atlaskv.utils.data_utils import generate_multi_entity_qa
from atlaskv.utils.eval_utils import (
    instruction_prompts,
    instruction_prompts_multi_entities,
    model_prune_format_mapping,
    zero_shot_prompt,
    zero_shot_prompt_multi_entities,
    _format_Q_llama,
    _format_Q_phi3,
    answer_question,
    softmax,
)
from atlaskv.utils.train_utils import get_hierarchical_kg_embd, get_kb_embd


# keep external behavior
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HOME", "your_HF_HOME")
os.environ.setdefault("TRANSFORMERS_CACHE", "your_TRANSFORMERS_CACHE")

nltk.download("wordnet")
logging.set_verbosity_warning()

ROUGE = evaluate.load("rouge")
BERT_SCORE = evaluate.load("bertscore")


class KBIndex:
    def __init__(
        self,
        encoder: KBEncoder,
        rows: List[Dict[str, Any]],
        key_path: Optional[str] = None,
        value_path: Optional[str] = None,
    ) -> None:
        self.encoder = encoder
        self.rows = rows
        self._keys = np.load(key_path).astype("float32") if key_path else None
        self._vals = np.load(value_path).astype("float32") if value_path else None
        if self._keys is not None:
            assert len(self.rows) == len(self._keys)

    def use_cache(self) -> bool:
        return self._keys is not None and self._vals is not None

    def encode(self, idx: Iterable[int]) -> Tuple[np.ndarray, np.ndarray]:
        if self.use_cache():
            return get_kb_embd(self.encoder, idx, precomputed_embd=(self._keys, self._vals))
        return get_kb_embd(self.encoder, idx, kb_dict=self.rows)


class KGIndex:
    def __init__(
        self,
        encoder: KBEncoder,
        rows: List[Dict[str, Any]],
        leaf_key_path: Optional[str] = None,
        value_path: Optional[str] = None,
        root_key_path: Optional[str] = None,
        inter_key_path: Optional[str] = None,
        root_c2id_path: Optional[str] = None,
        inter_c2id_path: Optional[str] = None,
        root_id2c_path: Optional[str] = None,
        inter_id2c_path: Optional[str] = None,
    ) -> None:
        self.encoder = encoder
        self.rows = rows
        self.leaf_keys = np.load(leaf_key_path).astype("float32") if leaf_key_path else None
        self.values = np.load(value_path).astype("float32") if value_path else None
        self.root_keys = np.load(root_key_path).astype("float32") if root_key_path else None
        self.inter_keys = np.load(inter_key_path).astype("float32") if inter_key_path else None
        self.root_c2id = json.load(open(root_c2id_path)) if root_c2id_path else None
        self.inter_c2id = json.load(open(inter_c2id_path)) if inter_c2id_path else None
        self.root_id2c = json.load(open(root_id2c_path)) if root_id2c_path else None
        self.inter_id2c = json.load(open(inter_id2c_path)) if inter_id2c_path else None
        if self.leaf_keys is not None:
            assert len(self.rows) == len(self.leaf_keys)

    def use_cache(self) -> bool:
        return self.leaf_keys is not None and self.values is not None

    def encode_hier(self, idx: Iterable[int]):
        if self.use_cache():
            return get_hierarchical_kg_embd(
                self.encoder,
                idx,
                self.root_id2c,
                self.inter_id2c,
                self.root_c2id,
                self.inter_c2id,
                mode="eval",
                precomputed_embd=(self.leaf_keys, self.inter_keys, self.root_keys, self.values),
            )
        return get_hierarchical_kg_embd(
            self.encoder,
            idx,
            self.root_id2c,
            self.inter_id2c,
            self.root_c2id,
            self.inter_c2id,
            mode="eval",
            kb_dict=self.rows,
        )


def _init_models(
    encoder_spec: str,
    encoder_ckpt: str,
    llm_type: str,
    llm_dir: str,
    model_dir: str,
    query_head_path: str,
    kb_layer_frequency: int,
    kb_scale_factor: Optional[int],
    use_kg: bool,
    encoding_batch_size: int,
    root_top_k_kb: int,
    inter_top_k_kb: int,
    leaf_top_k_kb: int,
    use_hierarchial_kv: bool,
):
    tok = AutoTokenizer.from_pretrained(
        llm_dir, trust_remote_code=True, padding_side="left", cache_dir="your_cache_dir"
    )
    tok.pad_token = "^"

    if llm_type == "llama3":
        ctor = AtlaskvLlamaForCausalLM if use_kg else KblamLlamaForCausalLM
        mdl = ctor.from_pretrained(
            model_dir, device_map="cuda", torch_dtype="auto", trust_remote_code=True
        )
        if query_head_path:
            mdl.load_query_head(query_head_path)
    else:
        mdl = KBLaMPhi3ForCausalLM.from_pretrained(
            model_dir, device_map="cuda", torch_dtype="auto", trust_remote_code=True
        )

    mdl.generation_config.pad_token_id = tok.pad_token_id
    mdl.generation_config.eos_token_id = tok.eos_token_id
    mdl.eval()

    if use_kg:
        kb_cfg = AtlasKVConfig(
            sep_query_head=True,
            kb_layer_frequency=kb_layer_frequency,
            kb_scale_factor=kb_scale_factor,
            root_top_k_kb=root_top_k_kb,
            inter_top_k_kb=inter_top_k_kb,
            leaf_top_k_kb=leaf_top_k_kb,
            use_hierarchial_kv=use_hierarchial_kv,
        )
    else:
        kb_cfg = KBLaMConfig(
            sep_query_head=True,
            kb_layer_frequency=kb_layer_frequency,
            kb_scale_factor=kb_scale_factor,
            use_hierarchial_kv=False,
        )

    enc = KBEncoder(
        encoder_name=encoder_spec,
        projector_type="linear",
        endpoint_url="your_endpoint_url",
        endpoint_api_key="your_endpoint_api_key",
        out_dim=mdl.config.hidden_size * (mdl.config.num_hidden_layers // kb_layer_frequency + 1),
        frozen_base_model=True,
        projector_kwargs={"mlp_depth": 1, "mlp_hidden_dim": 512},
        device=torch.device("cuda"),
        get_oai_embd_online=True if encoder_spec == "OAI" else False,
        encoding_batch_size=encoding_batch_size,
    )
    enc.load_state_dict(torch.load(encoder_ckpt))
    return tok, enc, mdl, kb_cfg


def _write_json(data: Dict[str, Any], path: Path) -> None:
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, sort_keys=True, default=str)
    except Exception as e:  # noqa: BLE001
        print(f"Error writing JSON file: {e}")


def _build_prompt_pairs(rows: List[Dict[str, Any]]) -> str:
    pieces = [f"{r['key_string']} is {r['description']}; " for r in rows]
    return "".join(pieces)


def _eval_loop(
    mdl,
    tok,
    kb_pack,
    kb_cfg,
    eval_mode: str,
    items: List[Dict[str, Any]],
    multi_entities: int,
    remove_sorry: bool,
    encoder_model_spec: str,
):
    torch.cuda.reset_peak_memory_stats()
    generated, gold, pairs = [], [], []

    if eval_mode == "icl":
        instr = instruction_prompts_multi_entities if multi_entities != -1 else instruction_prompts
        prompt_ctx = _build_prompt_pairs(items)
    elif eval_mode == "zeroshot":
        instr = zero_shot_prompt_multi_entities if multi_entities != -1 else zero_shot_prompt
        prompt_ctx = ""
    else:
        instr = ""
        prompt_ctx = ""

    for row in tqdm(items):
        if multi_entities == -1:
            q_text = row["Q"]
            gt_text = row["A"]
        else:
            sel = np.random.randint(0, len(items), multi_entities)
            q_text, gt_text = generate_multi_entity_qa(
                [items[i]["name"] for i in sel],
                [items[i]["description_type"] for i in sel],
                [items[i]["description"] for i in sel],
            )

        if eval_mode == "kb":
            out = answer_question(tok, mdl, q_text, kb=kb_pack, kb_config=kb_cfg).split(q_text)[1]
        elif eval_mode == "icl":
            out = answer_question(tok, mdl, instr + prompt_ctx + q_text, kb=None, kb_config=kb_cfg).split(q_text)[1]
        else:
            out = answer_question(tok, mdl, instr + q_text, kb=None, kb_config=kb_cfg).split(q_text)[1]

        if remove_sorry and "sorry" in out:
            continue

        pairs.append((out, gt_text))
        if multi_entities == -1:
            m = re.search(r'The\s+\w+\s+of\s+[^"]+\s+is\s+(.+)', out)
            gold.append(row["description"])
            if m:
                out = m.group(1)
        else:
            ms = re.findall(r"(?:is|are) (.*?)(?:\.|;)", out)
            out = "; ".join(ms)
            gold.append(";".join(re.findall(r"(?:is|are) (.*?);", gt_text)))

        generated.append(out)

    max_mem = torch.cuda.max_memory_reserved("cuda")
    for p, g in zip(generated, gold):
        print(f"True KB: {p}")
        print(f"True answer: {g}")
        print("--------------------")

    rouge_scores = ROUGE.compute(predictions=generated, references=gold)
    print(rouge_scores)
    res = {k: float(v) for k, v in rouge_scores.items()}

    bs = BERT_SCORE.compute(
        predictions=generated, references=gold, lang="en", model_type="microsoft/deberta-xlarge-mnli"
    )
    for k, v in bs.items():
        if isinstance(v, list):
            res[f"bert_score_{k}"] = float(np.mean(v))
            print(k, np.mean(v))

    transcript = "".join([f"Model output: {a}\nTrue answer: {b}\n-------\n" for a, b in pairs])
    tag = encoder_model_spec + "kb" if eval_mode == "kb" else eval_mode
    return transcript, res, max_mem, tag


def cmd_generation(args: argparse.Namespace) -> None:
    set_llama_attention_classes(args.use_kg)
    data = json.load(open(os.path.join(args.dataset_dir, args.test_dataset)))

    tok, enc, mdl, kb_cfg = _init_models(
        args.encoder_spec,
        args.encoder_dir,
        args.llm_type,
        args.llm_base_dir,
        args.model_dir,
        args.query_head_path,
        args.kb_layer_frequency,
        args.kb_scale_factor,
        args.use_kg,
        args.encoding_batch_size,
        args.root_top_k_kb,
        args.inter_top_k_kb,
        args.leaf_top_k_kb,
        args.use_hierarchial_kv,
    )

    if args.use_kg:
        kg = KGIndex(
            enc,
            data,
            leaf_key_path=args.precomputed_embed_keys_path,
            value_path=args.precomputed_embed_values_path,
            root_key_path=args.precomputed_embed_root_keys_path,
            inter_key_path=args.precomputed_embed_inter_keys_path,
            root_c2id_path=args.precomputed_embed_root_c2id_mapping_path,
            inter_c2id_path=args.precomputed_embed_inter_c2id_mapping_path,
            root_id2c_path=args.precomputed_embed_root_id2c_mapping_path,
            inter_id2c_path=args.precomputed_embed_inter_id2c_mapping_path,
        )
        kb = KBIndex(enc, data, args.precomputed_embed_keys_path, args.precomputed_embed_values_path)
    else:
        kg = None
        kb = KBIndex(enc, data, args.precomputed_embed_keys_path, args.precomputed_embed_values_path)

    rng = np.random.default_rng(args.seed)
    if len(data) <= args.kb_size:
        take_idx = np.arange(len(data))
    else:
        take_idx = rng.integers(0, len(data), size=args.kb_size)
    subset = [data[i] for i in take_idx]

    with torch.no_grad():
        if args.use_kg and kb_cfg.use_hierarchial_kv:
            kb_pack = kg.encode_hier(take_idx)  # type: ignore[arg-type]
        elif args.use_kg:
            kb_pack = kb.encode(take_idx)
        else:
            kb_pack = kb.encode(take_idx)

    transcript, scores, mem, tag = _eval_loop(
        mdl,
        tok,
        kb_pack,
        kb_cfg,
        args.eval_mode,
        subset[: min(args.sample_size, len(subset))],
        args.multi_entites,
        args.remove_sorry,
        args.encoder_spec,
    )

    scores["mem_cost"] = mem
    out_dir = Path(args.save_dir) / args.exp_config_name
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(scores, out_dir / f"{args.exp_config_name}.json")
    with (out_dir / f"{args.exp_config_name}.txt").open("w") as f:
        f.write(transcript)


def cmd_standard(args: argparse.Namespace) -> None:
    set_llama_attention_classes(args.use_kg)

    if args.kb_size == -1:
        kb_size = None
    else:
        kb_size = args.kb_size

    data = json.load(open(os.path.join(args.dataset_dir, args.test_dataset)))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.environ["ATTN_SAVE_DIR"] = args.save_dir
    os.environ["EVAL_MODE"] = "1"

    tok, enc, mdl, kb_cfg = _init_models(
        args.encoder_spec,
        args.encoder_dir,
        args.llm_type,
        args.llm_base_dir,
        args.model_dir,
        args.query_head_path,
        args.kb_layer_frequency,
        args.kb_scale_factor,
        args.use_kg,
        args.encoding_batch_size,
        args.root_top_k_kb,
        args.inter_top_k_kb,
        args.leaf_top_k_kb,
        args.use_hierarchial_kv,
    )

    for p in mdl.parameters():
        p.requires_grad = False

    # rebuild encoder to match original shape expectations when using different frequency
    enc = KBEncoder(
        encoder_name=args.encoder_spec,
        projector_type="linear",
        endpoint_url="your_endpoint_url",
        endpoint_api_key="your_endpoint_api_key",
        out_dim=mdl.config.hidden_size * (mdl.config.num_hidden_layers // args.kb_layer_frequency + 1),
        frozen_base_model=True,
        device=torch.device("cuda"),
        get_oai_embd_online=True if args.encoder_spec == "OAI" else False,
    )
    enc.load_state_dict(torch.load(args.encoder_dir))

    if kb_cfg.use_hierarchial_kv:
        index = KGIndex(
            enc,
            data,
            leaf_key_path=args.precomputed_embed_keys_path,
            value_path=args.precomputed_embed_values_path,
            root_key_path=args.precomputed_embed_root_keys_path,
            inter_key_path=args.precomputed_embed_inter_keys_path,
            root_c2id_path=args.precomputed_embed_root_c2id_mapping_path,
            inter_c2id_path=args.precomputed_embed_inter_c2id_mapping_path,
            root_id2c_path=args.precomputed_embed_root_id2c_mapping_path,
            inter_id2c_path=args.precomputed_embed_inter_id2c_mapping_path,
        )
    else:
        index = KBIndex(enc, data, args.precomputed_embed_keys_path, args.precomputed_embed_values_path)

    fmt = {"llama3": _format_Q_llama, "phi3": _format_Q_phi3}[args.llm_type]
    cfg_tag = f"{args.exp_config_str}__kb_{args.subset_size}__seed_{args.seed}"

    ranges = [(i, i + 1) for i in range(32)]
    batch_actual_sizes_all: List[List[int]] = []

    dataset_subset_idx = np.random.choice(len(data), args.subset_size, replace=False)
    dataset_subset = [data[i] for i in dataset_subset_idx]
    with torch.no_grad():
        if kb_cfg.use_hierarchial_kv:
            pack = index.encode_hier(dataset_subset_idx)
            (
                kb_root,
                kb_inter,
                kb_leaf,
                kb_val,
                root_idx,
                inter_idx,
                leaf_idx,
                root_c2id,
                inter_c2id,
            ) = pack
        else:
            kb_key, kb_val = index.encode(dataset_subset_idx)

    input_strs = [fmt(row["Q"]) for row in dataset_subset]
    answers = [row["A"] for row in dataset_subset]

    prune_str = None
    for m in model_prune_format_mapping:
        if isinstance(mdl, m):
            prune_str = model_prune_format_mapping[m]
            break
    if prune_str is None:
        raise ValueError("No prune_str found for this model type")

    num_batches = int(np.ceil(args.subset_size / args.batch_size))
    all_outputs_no_kb: List[str] = []
    all_outputs_true_kb: List[str] = []
    batch_actual_sizes: List[int] = []
    sample_num = min(args.sample_size, num_batches)
    for batch_idx in tqdm(range(num_batches)[:sample_num]):
        start = batch_idx * args.batch_size
        end = min((batch_idx + 1) * args.batch_size, args.subset_size)
        s_inputs = input_strs[start:end]
        if kb_cfg.use_hierarchial_kv:
            kb_tuple = (
                kb_root,
                kb_inter,
                kb_leaf,
                kb_val,
                root_idx,
                inter_idx,
                leaf_idx,
                root_c2id,
                inter_c2id,
            )
        else:
            kb_tuple = (kb_key, kb_val)
        tokenized = tok(s_inputs, return_tensors="pt", padding=True).to("cuda")
        inp_ids = tokenized["input_ids"]
        attn_mask = tokenized["attention_mask"]
        batch_actual_sizes.append(end - start)

        base_name = f"{cfg_tag}_batch{batch_idx}"
        with torch.no_grad():
            out_no_kb = mdl.generate(
                input_ids=inp_ids,
                attention_mask=attn_mask,
                kb_kvs=None,
                max_new_tokens=40,
                tokenizer=tok,
                output_attentions=False,
                kb_config=kb_cfg,
            )
            out_true_kb = mdl.generate(
                input_ids=inp_ids,
                attention_mask=attn_mask,
                kb_kvs=kb_tuple,
                max_new_tokens=40,
                tokenizer=tok,
                output_attentions=True,
                save_attention_weights=True,
                attention_save_loc=args.save_dir,
                attention_file_base_name=base_name,
                kb_config=kb_cfg,
            )
        all_outputs_no_kb.extend(tok.batch_decode(out_no_kb, skip_special_tokens=False))
        all_outputs_true_kb.extend(tok.batch_decode(out_true_kb, skip_special_tokens=False))

    batch_actual_sizes_all.append(batch_actual_sizes)

    no_kb_predictions, predictions, gold = [], [], []
    for i in range(args.subset_size)[: sample_num * args.batch_size]:
        print("True KB", prune_str(all_outputs_true_kb[i]))
        print("True answer: ", answers[i])
        no_kb_predictions.append(prune_str(all_outputs_no_kb[i]))
        predictions.append(prune_str(all_outputs_true_kb[i]))
        gold.append(answers[i])
        print("--------------------")

    # aggregate attention
    accs_to_save: List[List[float]] = []
    confidences_to_save: List[float] = []
    for left, right in ranges:
        batch_accs: List[float] = []
        batch_top5_accs: List[float] = []
        batch_confidences: List[float] = []
        batch_sizes_this_round = batch_actual_sizes_all[-1][:sample_num]
        for b_idx, bsize in enumerate(batch_sizes_this_round):
            base_name = f"{cfg_tag}_batch{b_idx}"
            per_layer = []
            for layer_idx in range(32)[left:right]:
                if layer_idx % args.kb_layer_frequency == 0:
                    pth = os.path.join(args.save_dir, f"{base_name}_{layer_idx}.npy")
                    if not os.path.exists(pth):
                        continue
                    w = np.load(pth)
                    w = w[..., : args.subset_size]
                    w = w.reshape(bsize, -1, args.subset_size)
                    per_layer.append(w)
            if not per_layer:
                continue
            w_all = np.stack(per_layer, axis=1).reshape(bsize, -1, args.subset_size)
            start = b_idx * args.batch_size
            end = start + bsize
            gt = np.arange(args.subset_size)[start:end]
            acc = (w_all.sum(1).argmax(1) == gt).mean()
            top5 = torch.topk(torch.from_numpy(w_all.sum(1)), 5, dim=1)[1]
            top5_acc = ((top5.numpy() == gt[:, None]).any(1)).mean()
            conf = softmax(w_all.mean(1), -1).max()
            batch_accs.append(float(acc))
            batch_top5_accs.append(float(top5_acc))
            batch_confidences.append(float(conf))
        if batch_accs:
            accs_to_save.append([float(np.mean(batch_accs)), float(np.mean(batch_top5_accs))])
            confidences_to_save.append(float(np.mean(batch_confidences)))
        else:
            accs_to_save.append([0.0, 0.0])
            confidences_to_save.append(0.0)

    accs_np = np.array(accs_to_save)
    conf_np = np.array(confidences_to_save)
    np.save(os.path.join(args.attn_summary_save_dir, f"{cfg_tag}_acc.npy"), accs_np)
    np.save(os.path.join(args.attn_summary_save_dir, f"{cfg_tag}_conf.npy"), conf_np)

    rouge_pred = ROUGE.compute(predictions=predictions, references=gold)
    np.savez(os.path.join(args.attn_summary_save_dir, f"{cfg_tag}_rouge.npy"), **rouge_pred)

    rouge_no_kb = ROUGE.compute(predictions=no_kb_predictions, references=gold)
    np.savez(
        os.path.join(args.attn_summary_save_dir, f"{cfg_tag}_rouge_no_kb.npy"),
        **rouge_no_kb,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Alternative evaluation script")

    # shared
    shared = argparse.ArgumentParser(add_help=False)
    shared_args = [
        ("--dataset_dir", dict(type=str, help="Directory containing the dataset")),
        ("--encoder_dir", dict(type=str, help="Directory containing the encoder model")),
        ("--encoder_spec", dict(type=str, default="OAI", help="Specification for the encoder model")),
        ("--fancy_instruction", dict(action=argparse.BooleanOptionalAction, default=False, help="Whether to use fancy instructions")),
        ("--kb_layer_frequency", dict(type=int, default=3, help="Frequency of knowledge base layers")),
        ("--kb_scale_factor", dict(type=int, default=None, help="Scaling factor for knowledge base")),
        ("--kb_size", dict(type=int, default=200, help="Size of the knowledge base")),
        ("--llm_base_dir", dict(type=str, help="llm to load, can be HF location or local directory")),
        ("--llm_type", dict(type=str, default="phi3", choices=["llama3", "phi3"], help="Type of language model to use")),
        ("--model_dir", dict(type=str, help="Directory containing the model")),
        ("--save_dir", dict(type=str, help="Directory to save outputs")),
        ("--seed", dict(type=int, help="Random seed for reproducibility")),
        ("--test_dataset", dict(type=str, help="Source of test KB (assumes KV pair format)")),
        ("--precomputed_embed_keys_path", dict(type=str, help="Path to precomputed key embeddings")),
        ("--precomputed_embed_values_path", dict(type=str, help="Path to precomputed value embeddings")),
        ("--encoding_batch_size", dict(type=int, default=64, help="Batch size for encoding keys and values")),
        ("--precomputed_embed_root_keys_path", dict(type=str, help="Path to precomputed root keys embeddings")),
        ("--precomputed_embed_inter_keys_path", dict(type=str, help="Path to precomputed inter keys embeddings")),
        ("--precomputed_embed_root_c2id_mapping_path", dict(type=str, help="Path to precomputed root c2id mapping")),
        ("--precomputed_embed_inter_c2id_mapping_path", dict(type=str, help="Path to precomputed inter c2id mapping")),
        ("--precomputed_embed_root_id2c_mapping_path", dict(type=str, help="Path to precomputed root id2c mapping")),
        ("--precomputed_embed_inter_id2c_mapping_path", dict(type=str, help="Path to precomputed inter id2c mapping")),
        ("--query_head_path", dict(type=str, default="", help="Path to load KB head from")),
        ("--use_kg", dict(action="store_true")),
    ]
    for flag, kwargs in shared_args:
        shared.add_argument(flag, **kwargs)

    sub = parser.add_subparsers(dest="command", required=True)

    # generation
    g = sub.add_parser("generation", parents=[shared], help="Evaluate generation")
    g_args = [
        ("--eval_mode", dict(type=str, choices=["kb", "icl", "zeroshot"], default="kb")),
        ("--exp_config_name", dict(type=str, default="generation_results")),
        ("--kb_token_layer_frequency", dict(type=int, default=None)),
        ("--multi_entites", dict(type=int, default=-1)),
        ("--no_outlier", dict(action=argparse.BooleanOptionalAction, default=False)),
        ("--remove_sorry", dict(action=argparse.BooleanOptionalAction, default=False)),
        ("--topk_size", dict(type=int, default=-1)),
        ("--root_top_k_kb", dict(type=int, default=128)),
        ("--inter_top_k_kb", dict(type=int, default=64)),
        ("--leaf_top_k_kb", dict(type=int, default=16)),
        ("--use_hierarchial_kv", dict(action=argparse.BooleanOptionalAction, default=False)),
        ("--sample_size", dict(type=int, default=50)),
    ]
    for flag, kwargs in g_args:
        g.add_argument(flag, **kwargs)

    # standard
    s = sub.add_parser("standard", parents=[shared], help="Evaluate basic performance")
    s_args = [
        ("--attn_summary_save_dir", dict(type=str, default="")),
        ("--eval_mode", dict(type=str, choices=["kb", "icl", "zeroshot"], default="kb")),
        ("--exp_config_name", dict(type=str, default="basic_results")),
        ("--exp_config_str", dict(type=str)),
        ("--kb_token_layer_frequency", dict(type=int, default=None)),
        ("--no_outlier", dict(action=argparse.BooleanOptionalAction, default=False)),
        ("--sample_size", dict(type=int, default=50)),
        ("--subset_size", dict(type=int, default=100)),
        ("--topk_size", dict(type=int, default=-1)),
        ("--batch_size", dict(type=int, default=1)),
        ("--root_top_k_kb", dict(type=int, default=128)),
        ("--inter_top_k_kb", dict(type=int, default=64)),
        ("--leaf_top_k_kb", dict(type=int, default=16)),
        ("--use_hierarchial_kv", dict(action=argparse.BooleanOptionalAction, default=False)),
    ]
    for flag, kwargs in s_args:
        s.add_argument(flag, **kwargs)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(args)
    if args.command == "generation":
        cmd_generation(args)
    elif args.command == "standard":
        cmd_standard(args)
    else:
        raise ValueError(f"command {args.command} not recognised")


if __name__ == "__main__":
    main()


