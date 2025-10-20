import argparse
import json
import logging
import os
import pathlib
import re
from dataclasses import dataclass
from functools import partial
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import transformers
import wandb
from accelerate import Accelerator
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.theme import Theme
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer

from atlaskv.kb_encoder import KBEncoder
from atlaskv.models.kblam_config import AtlasKVConfig, KBLaMConfig
from atlaskv.models.llama3_model import (
    AtlaskvLlamaForCausalLM,
    KblamLlamaForCausalLM,
    set_llama_attention_classes,
)
from atlaskv.models.phi3_model import KBLaMPhi3ForCausalLM
from atlaskv.utils.data_utils import (
    augment_row,
    generate_multi_entity_qa,
    get_i_dont_know_ans,
)
from atlaskv.utils.train_utils import (
    context_set_size_scheduler,
    get_hierarchical_kg_embd,
    get_kb_embd,
    setup_scheduler_and_optimizer,
)

# Environment setup
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("NCCL_TIMEOUT", "1200000")

# Logging configuration
LOGFORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOGFORMAT_RICH = "%(message)s"

custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow", 
    "error": "bold red",
    "critical": "bold white on red",
})

console = Console(theme=custom_theme)

logging.basicConfig(
    level=logging.WARNING,
    format=LOGFORMAT_RICH,
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    seed: int = 1
    train_dataset: str = "synthetic_data_qkv"
    N: int = 120000
    B: int = 10
    lr: float = 1e-4
    total_steps: int = 20000
    encoder_spec: str = "OAI"
    key_embd_src: str = "key"
    gradient_accm_step: int = 20
    kb_token_layer_frequency: int = 3
    llm_type: str = "llama3"
    hf_model_spec: str = "unsloth/Meta-Llama-3.1-8B-Instruct"
    max_seq_len: Optional[int] = None
    kb_size: Optional[int] = None
    dynamic_kb_size: Optional[Tuple[int, int]] = None
    outlier_num: int = 1
    multi_entities: Optional[int] = None
    sep_query_head: bool = False
    use_data_aug: bool = False
    use_lr_decay: bool = False
    use_cached_embd: bool = False
    use_certainty_loss: bool = False
    use_kg: bool = False
    use_hierarchial_kv: bool = False
    use_extended_qa: bool = False
    duplicate_true_kb: bool = True
    length_invariance: bool = False
    projector_type: str = "linear"
    verbose: bool = False
    log_to_file: bool = False


class DataProcessor:
    """Handles data loading and preprocessing."""
    
    @staticmethod
    def load_cached_embeddings(
        encoder_spec: str, 
        dataset_dir: str, 
        dataset_name: str, 
        key_embd_src: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load precomputed embeddings."""
        spec_str = "OAI" if encoder_spec == "OAI" else encoder_spec
        key_path = os.path.join(
            dataset_dir, f"{dataset_name}_{spec_str}_embd_{key_embd_src}.npy"
        )
        value_src = "answer" if key_embd_src == "answer" else "value"
        value_path = os.path.join(
            dataset_dir, f"{dataset_name}_{spec_str}_embd_{value_src}.npy"
        )
        return np.load(key_path).astype("float32"), np.load(value_path).astype("float32")

    @staticmethod
    def load_cached_hierarchical_embeddings(
        encoder_spec: str,
        dataset_dir: str, 
        dataset_name: str,
        key_embd_src: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict, Dict, Dict, Dict]:
        """Load hierarchical embeddings and mappings."""
        spec_str = "OAI" if encoder_spec == "OAI" else encoder_spec
        
        # Load embeddings
        leaf_keys = np.load(os.path.join(
            dataset_dir, f"{dataset_name}_{spec_str}_embd_{key_embd_src}.npy"
        )).astype("float32")
        inter_keys = np.load(os.path.join(
            dataset_dir, f"{dataset_name}_{spec_str}_embd_{key_embd_src}_inter1.npy"
        )).astype("float32")
        root_keys = np.load(os.path.join(
            dataset_dir, f"{dataset_name}_{spec_str}_embd_{key_embd_src}_root.npy"
        )).astype("float32")
        
        value_src = "answer" if key_embd_src == "answer" else "value"
        values = np.load(os.path.join(
            dataset_dir, f"{dataset_name}_{spec_str}_embd_{value_src}.npy"
        )).astype("float32")
        
        # Load mappings
        mapping_paths = {
            "root_c2id": f"{dataset_name}_{spec_str}_embd_{key_embd_src}_root_c2id_mapping.json",
            "root_id2c": f"{dataset_name}_{spec_str}_embd_{key_embd_src}_root_id2c_mapping.json", 
            "inter_c2id": f"{dataset_name}_{spec_str}_embd_{key_embd_src}_inter1_c2id_mapping.json",
            "inter_id2c": f"{dataset_name}_{spec_str}_embd_{key_embd_src}_inter1_id2c_mapping.json",
        }
        
        mappings = {}
        for key, filename in mapping_paths.items():
            with open(os.path.join(dataset_dir, filename)) as f:
                mappings[key] = json.load(f)
        
        return leaf_keys, inter_keys, root_keys, values, mappings["root_c2id"], mappings["inter_c2id"], mappings["root_id2c"], mappings["inter_id2c"]

    @staticmethod
    def load_dataset(dataset_dir: str, dataset_name: str, use_extended: bool = False) -> List[Dict]:
        """Load training dataset."""
        if use_extended:
            try:
                return json.load(open(os.path.join(dataset_dir, f"{dataset_name}_augmented.json")))
            except:
                return json.load(open(os.path.join(dataset_dir, f"{dataset_name}_augmented.jsonl")))
        else:
            try:
                return json.load(open(os.path.join(dataset_dir, f"{dataset_name}.json")))
            except:
                return json.load(open(os.path.join(dataset_dir, f"{dataset_name}.jsonl")))


class ModelFactory:
    """Factory for creating models and tokenizers."""
    
    @staticmethod
    def create_tokenizer(hf_model_spec: str, hf_token: Optional[str], llm_type: str) -> AutoTokenizer:
        """Create and configure tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_spec,
            trust_remote_code=True,
            token=hf_token if hf_token and llm_type == "llama3" else None,
            cache_dir='your_cache_dir',
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @staticmethod
    def create_model(
        model_spec: str,
        llm_type: str,
        use_kg: bool,
        device: torch.device,
        hf_token: Optional[str] = None
    ) -> Union[KBLaMPhi3ForCausalLM, KblamLlamaForCausalLM, AtlaskvLlamaForCausalLM]:
        """Create and configure model."""
        if llm_type == "llama3":
            if use_kg:
                model = AtlaskvLlamaForCausalLM.from_pretrained(
                    model_spec,
                    device_map=device,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    token=hf_token,
                    cache_dir='your_cache_dir'
                )
            else:
                model = KblamLlamaForCausalLM.from_pretrained(
                    model_spec,
                    device_map=device,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    token=hf_token,
                    cache_dir='your_cache_dir'
                )
        elif llm_type == "phi3":
            if use_kg:
                raise ValueError("phi3 is currently not supported for AtlasKV.")
            model = KBLaMPhi3ForCausalLM.from_pretrained(
                model_spec,
                device_map=device,
                torch_dtype="auto",
                trust_remote_code=True,
                cache_dir='your_cache_dir'
            )
        else:
            raise ValueError(f"LLM type {llm_type} not recognised")
        
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    @staticmethod
    def create_encoder(
        encoder_spec: str,
        model_config: Any,
        kb_token_layer_frequency: int,
        projector_type: str,
        device: torch.device
    ) -> KBEncoder:
        """Create and configure encoder."""
        return KBEncoder(
            encoder_name=encoder_spec,
            projector_type=projector_type,
            endpoint_url="https://yunwu.ai/v1",
            endpoint_api_key="sk-7IQRBZIKqwifDeQEJIyG7twNFcR4MIzXzbqTbd26sUD4azDt",
            out_dim=model_config.hidden_size * (model_config.num_hidden_layers // kb_token_layer_frequency + 1),
            frozen_base_model=True,
            projector_kwargs={"mlp_depth": 1, "mlp_hidden_dim": model_config.hidden_size},
            device=device,
            get_oai_embd_online=True if encoder_spec == "OAI" else False
        )


class DataRetriever:
    """Handles knowledge base retrieval during training."""
    
    def __init__(
        self,
        encoder: KBEncoder,
        dataset: List[Dict],
        key_embds: Optional[np.ndarray] = None,
        value_embds: Optional[np.ndarray] = None,
        leaf_key_embds: Optional[np.ndarray] = None,
        inter_key_embds: Optional[np.ndarray] = None,
        root_key_embds: Optional[np.ndarray] = None,
        root_c2id_mapping: Optional[Dict] = None,
        inter_c2id_mapping: Optional[Dict] = None,
        root_id2c_mapping: Optional[Dict] = None,
        inter_id2c_mapping: Optional[Dict] = None,
    ):
        self.encoder = encoder
        self.dataset = dataset
        self.key_embds = key_embds
        self.value_embds = value_embds
        self.leaf_key_embds = leaf_key_embds
        self.inter_key_embds = inter_key_embds
        self.root_key_embds = root_key_embds
        self.root_c2id_mapping = root_c2id_mapping
        self.inter_c2id_mapping = inter_c2id_mapping
        self.root_id2c_mapping = root_id2c_mapping
        self.inter_id2c_mapping = inter_id2c_mapping

    def _use_cached_embd(self) -> bool:
        """Check if cached embeddings are available."""
        if self.key_embds is not None and self.value_embds is not None:
            return True
        return False

    def _use_cached_hierarchical_embd(self) -> bool:
        """Check if cached hierarchical embeddings are available."""
        return all([
            self.leaf_key_embds is not None,
            self.inter_key_embds is not None, 
            self.root_key_embds is not None,
            self.value_embds is not None
        ])

    def get_embeddings(self, batch_indices: List[int], batch_size: int, step: int, kb_size: int) -> Tuple:
        """Get embeddings for training batch."""
        if self._use_cached_embd():
            train_key, train_val = get_kb_embd(
                self.encoder, batch_indices, precomputed_embd=(self.key_embds, self.value_embds)
            )
        else:
            train_key, train_val = get_kb_embd(self.encoder, batch_indices, kb_dict=self.dataset)

        if len(train_key.shape) == 2:
            train_key = train_key.unsqueeze(0).transpose(0, 1)
            train_val = train_val.unsqueeze(0).transpose(0, 1)

        context_size = context_set_size_scheduler(step, kb_size)
        context_indices = np.random.choice(len(self.dataset), context_size, replace=False)
        
        if self._use_cached_embd():
            context_key, context_val = get_kb_embd(
                self.encoder, context_indices, precomputed_embd=(self.key_embds, self.value_embds)
            )
        else:
            context_key, context_val = get_kb_embd(self.encoder, context_indices, kb_dict=self.dataset)
        
        context_key = context_key.unsqueeze(0).expand(batch_size, *context_key.shape)
        context_val = context_val.unsqueeze(0).expand(batch_size, *context_val.shape)
        
        return (
            torch.concat([train_key, context_key], 1),
            torch.concat([train_val, context_val], 1)
        )

    def get_hierarchical_embeddings(self, batch_indices: List[int], batch_size: int, step: int, kb_size: int) -> Tuple:
        """Get hierarchical embeddings for training batch."""
        if self._use_cached_hierarchical_embd():
            train_root, train_inter, train_leaf, train_val, train_root_idx, train_inter_idx, train_leaf_idx, root_c2id, inter_c2id = get_hierarchical_kg_embd(
                self.encoder, batch_indices, self.root_id2c_mapping, self.inter_id2c_mapping,
                self.root_c2id_mapping, self.inter_c2id_mapping, mode="train",
                precomputed_embd=(self.leaf_key_embds, self.inter_key_embds, self.root_key_embds, self.value_embds)
            )
        else:
            train_root, train_inter, train_leaf, train_val, train_root_idx, train_inter_idx, train_leaf_idx, root_c2id, inter_c2id = get_hierarchical_kg_embd(
                self.encoder, batch_indices, self.root_id2c_mapping, self.inter_id2c_mapping,
                self.root_c2id_mapping, self.inter_c2id_mapping, mode="train", kb_dict=self.dataset
            )

        # Reshape if needed
        if len(train_leaf) == 2:
            train_leaf = train_leaf.unsqueeze(0).transpose(0, 1)
            train_inter = train_inter.unsqueeze(0).transpose(0, 1)
            train_root = train_root.unsqueeze(0).transpose(0, 1)
            train_val = train_val.unsqueeze(0).transpose(0, 1)
            train_root_idx = train_root_idx.unsqueeze(0).transpose(0, 1)
            train_inter_idx = train_inter_idx.unsqueeze(0).transpose(0, 1)
            train_leaf_idx = train_leaf_idx.unsqueeze(0).transpose(0, 1)

        context_size = context_set_size_scheduler(step, kb_size)
        context_indices = np.random.choice(len(self.dataset), context_size, replace=False)
        
        if self._use_cached_hierarchical_embd():
            context_root, context_inter, context_leaf, context_val, context_root_idx, context_inter_idx, context_leaf_idx, _, _ = get_hierarchical_kg_embd(
                self.encoder, context_indices, self.root_id2c_mapping, self.inter_id2c_mapping,
                self.root_c2id_mapping, self.inter_c2id_mapping, mode="train",
                precomputed_embd=(self.leaf_key_embds, self.inter_key_embds, self.root_key_embds, self.value_embds)
            )
        else:
            context_root, context_inter, context_leaf, context_val, context_root_idx, context_inter_idx, context_leaf_idx, _, _ = get_hierarchical_kg_embd(
                self.encoder, context_indices, self.root_id2c_mapping, self.inter_id2c_mapping,
                self.root_c2id_mapping, self.inter_c2id_mapping, mode="train", kb_dict=self.dataset
            )

        # Expand context embeddings
        context_root = context_root.expand(batch_size, *context_root.shape).squeeze()
        context_inter = context_inter.expand(batch_size, *context_inter.shape).squeeze()
        context_leaf = context_leaf.expand(batch_size, *context_leaf.shape).squeeze()
        context_val = context_val.expand(batch_size, *context_val.shape).squeeze()
        context_root_idx = context_root_idx.expand(batch_size, *context_root_idx.shape).squeeze()
        context_inter_idx = context_inter_idx.expand(batch_size, *context_inter_idx.shape).squeeze()
        context_leaf_idx = context_leaf_idx.expand(batch_size, *context_leaf_idx.shape).squeeze()

        return (
            torch.concat([train_root, context_root], 1),
            torch.concat([train_inter, context_inter], 1),
            torch.concat([train_leaf, context_leaf], 1),
            torch.concat([train_val, context_val], 1),
            torch.concat([train_root_idx, context_root_idx], 1),
            torch.concat([train_inter_idx, context_inter_idx], 1),
            torch.concat([train_leaf_idx, context_leaf_idx], 1),
            root_c2id, inter_c2id
        )


class TrainingStepManager:
    """Manages training step configuration and data generation."""
    
    @staticmethod
    def get_step_config(
        current_step: int,
        total_steps: int,
        use_data_aug: bool,
        outlier_num: int,
        multi_entities: Optional[int],
        use_extended_qa: bool,
    ) -> Dict[str, Any]:
        """Get configuration for current training step."""
        config = {
            "use_data_aug": use_data_aug,
            "include_outlier": False,
            "multi_entities": None,
            "use_extended_qa": False,
        }
        
        include_outlier = current_step >= total_steps - 1 - outlier_num
        if include_outlier:
            config["include_outlier"] = True
            return config
        
        if current_step % 3 == 0:
            config["multi_entities"] = multi_entities
            return config
        if current_step % 3 == 1:
            config["use_extended_qa"] = use_extended_qa
            return config
        return config

    @staticmethod
    def get_batch(
        qa_format_func: Callable[[str, str], str],
        label_func: Callable[[torch.Tensor, List, Callable], torch.Tensor],
        dataset: List[Dict],
        tokenizer,
        device: torch.device,
        B: int = 20,
        random_sample: bool = True,
        use_data_aug: bool = False,
        include_outlier: bool = False,
        multi_entities: Optional[int] = None,
        use_extended_qa: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """Generate training batch."""
        if multi_entities is not None:
            assert not include_outlier

        if random_sample:
            if multi_entities is not None:
                batch_indices = np.random.choice(len(dataset), (B, multi_entities), replace=False)
            else:
                batch_indices = np.random.choice(len(dataset), B, replace=False)
        else:
            batch_indices = np.arange(B)

        def get_qa_pair(idx: int) -> Tuple[str, str]:
            if use_extended_qa:
                return dataset[idx]["extended_Q"], dataset[idx]["extended_A"]
            elif multi_entities is not None:
                return generate_multi_entity_qa(
                    [dataset[i]["name"] for i in idx],
                    [dataset[i]["description_type"] for i in idx],
                    [dataset[i]["description"] for i in idx],
                )
            else:
                Q = augment_row(dataset[idx]) if use_data_aug else dataset[idx]["Q"]
                A = get_i_dont_know_ans() if include_outlier else dataset[idx]["A"]
                return Q, A

        with torch.autograd.no_grad():
            input_strs = []
            real_batch_indices = []
            for idx in batch_indices:
                Q, A = get_qa_pair(idx)
                if Q is not None and A is not None:
                    input_strs.append(qa_format_func(Q, A))
                    real_batch_indices.append(idx)
                else:
                    print("Q or Answer is none")
            batch_indices = real_batch_indices
            tokenizer_output = tokenizer(input_strs, return_tensors="pt", padding=True).to(device)
            input_ids, attention_masks = (
                tokenizer_output["input_ids"],
                tokenizer_output["attention_mask"],
            )
            labels = label_func(input_ids, input_strs, tokenizer)
        
        if include_outlier:
            batch_indices = np.random.choice(len(dataset), B, replace=False)
        return input_ids, attention_masks, labels, batch_indices


class ModelTrainer:
    """Main training orchestrator."""
    
    def __init__(
        self,
        model: Union[KBLaMPhi3ForCausalLM, KblamLlamaForCausalLM, AtlaskvLlamaForCausalLM],
        retriever: DataRetriever,
        tokenizer: AutoTokenizer,
        config: TrainingConfig,
        device: torch.device,
        output_dir: str,
    ):
        self.accelerator = Accelerator(device_placement=False)
        self.logger = logging.getLogger("training")
        self.model = model
        self.retriever = retriever
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.output_dir = pathlib.Path(output_dir)
        
        # Setup model-specific functions
        if isinstance(model, KBLaMPhi3ForCausalLM):
            self._get_batch = partial(TrainingStepManager.get_batch, self._format_qa_phi3, self._create_labels_phi3)
            self._get_params = self._get_phi3_params
        elif isinstance(model, (KblamLlamaForCausalLM, AtlaskvLlamaForCausalLM)):
            self._get_batch = partial(TrainingStepManager.get_batch, self._format_qa_llama, self._create_labels_llama)
            self._get_params = self._get_llama3_params
        else:
            raise ValueError(f"{model} not recognised")

        self.scheduler, self.optim = self._setup_optimizer()
        self.model, self.optim, self._get_batch, self.retriever.encoder = self.accelerator.prepare(
            self.model, self.optim, self._get_batch, self.retriever.encoder
        )

    def _format_qa_llama(self, Q: str, A: str) -> str:
        """Format Q&A for Llama model."""
        return (
            "<|start_header_id|>user<|end_header_id|> " + Q + "<|eot_id|>"
            + "<|start_header_id|>assistant<|end_header_id|>" + A + "<|eot_id|>"
        )

    def _format_qa_phi3(self, Q: str, A: str) -> str:
        """Format Q&A for Phi3 model."""
        return "<|user|>\n" + Q + "<|end|>\n" + "<|assistant|>\n" + A + "<|end|>\n"

    def _create_labels_llama(self, input_ids: torch.Tensor, input_strs: List[str], tokenizer) -> torch.Tensor:
        """Create labels for Llama model."""
        answer_indices = torch.argmax(
            (input_ids == tokenizer("<|start_header_id|>assistant<|end_header_id|>")["input_ids"][2]).long(),
            -1,
        )
        answer_mask = torch.ones_like(input_ids)
        for b in range(len(input_strs)):
            answer_mask[b, : (answer_indices[b].item() + 2)] = 0
        return input_ids * answer_mask + (1 - answer_mask) * (-100)

    def _create_labels_phi3(self, input_ids: torch.Tensor, input_strs: List[str], tokenizer) -> torch.Tensor:
        """Create labels for Phi3 model."""
        answer_indices = torch.argmax(
            (input_ids == tokenizer("<|user|>")["input_ids"][0]).long(),
            -1,
        )
        answer_mask = torch.ones_like(input_ids)
        for b in range(len(input_strs)):
            answer_mask[b, : (answer_indices[b].item() + 1)] = 0
        return input_ids * answer_mask + (1 - answer_mask) * (-100)

    def _get_phi3_params(self, model, sep_query_head: bool, kb_token_layer_frequency: int) -> List[torch.nn.Parameter]:
        """Get trainable parameters for Phi3 model."""
        params = []
        for name, param in model.named_parameters():
            if sep_query_head:
                if "qkv_proj.weight" in name:
                    layer_id = int(re.search(r"\d+", name)[0])
                    if layer_id % kb_token_layer_frequency == 0:
                        old_weight = param.detach()
                if "q_proj_new.weight" in name:
                    layer_id = int(re.search(r"\d+", name)[0])
                    if layer_id % kb_token_layer_frequency == 0:
                        param.copy_(old_weight[: model.config.hidden_size, :])
                        param.requires_grad = True
                        params.append(param)
                if "q_proj_kg.weight" in name:
                    layer_id = int(re.search(r"\d+", name)[0])
                    if layer_id % kb_token_layer_frequency == 0:
                        param.copy_(old_weight[: model.config.hidden_size, :])
                        param.requires_grad = True
                        params.append(param)
            else:
                if "q_proj.weight" in name:
                    layer_id = int(re.search(r"\d+", name)[0])
                    if layer_id % kb_token_layer_frequency == 0:
                        param.requires_grad = True
                        params.append(param)
        return params

    def _get_llama3_params(self, model, sep_query_head: bool, kb_token_layer_frequency: int) -> List[torch.nn.Parameter]:
        """Get trainable parameters for Llama3 model."""
        params = []
        for name, param in model.named_parameters():
            if sep_query_head:
                if "q_proj.weight" in name:
                    layer_id = int(re.search(r"\d+", name)[0])
                    if layer_id % kb_token_layer_frequency == 0:
                        old_weight = param.detach()
                if "q_proj_new.weight" in name:
                    layer_id = int(re.search(r"\d+", name)[0])
                    if layer_id % kb_token_layer_frequency == 0:
                        param.copy_(old_weight)
                        param.requires_grad = True
                        params.append(param)
                if "q_proj_kg.weight" in name:
                    layer_id = int(re.search(r"\d+", name)[0])
                    if layer_id % kb_token_layer_frequency == 0:
                        param.copy_(old_weight)
                        param.requires_grad = True
                        params.append(param)
            else:
                if "q_proj.weight" in name:
                    layer_id = int(re.search(r"\d+", name)[0])
                    if layer_id % kb_token_layer_frequency == 0:
                        param.requires_grad = True
                        params.append(param)
        return params

    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        if self.config.sep_query_head:
            self.logger.info("Query head being fine tuned!")
            llm_q_params = self._get_params(self.model, self.config.sep_query_head, self.config.kb_token_layer_frequency)
            scheduler, optim = setup_scheduler_and_optimizer(
                chain(self.retriever.encoder.parameters(), llm_q_params),
                self.config.lr,
                self.config.total_steps,
            )
        else:
            scheduler, optim = setup_scheduler_and_optimizer(
                self.retriever.encoder.parameters(), self.config.lr, self.config.total_steps
            )
        return scheduler, optim

    def train(
        self,
        training_set: List[Dict],
        batch_size: int,
        grad_accum_steps: int,
        outlier_num: int,
        use_data_aug: bool = False,
        multi_entities: Optional[int] = None,
        use_extended_qa: bool = False,
        save_period: int = 2000,
        resumed_step: int = 0,
        kb_config: Union[KBLaMConfig, AtlasKVConfig] = None,
    ):
        """Main training loop."""
        train_losses = []
        start_step = resumed_step
        loss_fct = CrossEntropyLoss(reduction="none")

        num_processes = self.accelerator.num_processes
        accum_steps_per_gpu = max(1, grad_accum_steps // num_processes)

        if self.accelerator.is_main_process:
            self.logger.info(f"Training with {num_processes} GPUs")
            self.logger.info(f"Total accumulation steps: {grad_accum_steps}, Steps per GPU: {accum_steps_per_gpu}")

        with self._create_progress_bar() as pbar:
            task = pbar.add_task("Training", total=self.config.total_steps, loss=100)
            for step in range(start_step, self.config.total_steps, 1):
                self.optim.zero_grad()
                losses = []

                process_rank = self.accelerator.process_index
                start_accum_step = process_rank * accum_steps_per_gpu
                end_accum_step = min(start_accum_step + accum_steps_per_gpu, grad_accum_steps)

                for a_step in range(start_accum_step, end_accum_step):
                    step_config = TrainingStepManager.get_step_config(
                        a_step, grad_accum_steps, use_data_aug, outlier_num, multi_entities, use_extended_qa
                    )
                    input_ids, attention_masks, labels, batch_indices = self._get_batch(
                        training_set, self.tokenizer, self.device, B=batch_size, random_sample=True, **step_config
                    )

                    if a_step == 0 and step % 10 == 0:
                        self.logger.info(f"INPUT IDs SHAPE: {input_ids.shape}")

                    if self.config.max_seq_len is not None:
                        input_ids = input_ids[:, : self.config.max_seq_len]
                        attention_masks = attention_masks[:, : self.config.max_seq_len]
                        labels = labels[:, : self.config.max_seq_len]

                    if self.config.use_hierarchial_kv:
                        kb_embedding = self.retriever.get_hierarchical_embeddings(
                            batch_indices, len(input_ids), step, self.config.kb_size
                        )
                    else:
                        kb_embedding = self.retriever.get_embeddings(
                            batch_indices, len(input_ids), step, self.config.kb_size
                        )

                    out = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_masks,
                        kb_kvs=kb_embedding,
                        output_attentions=True,
                        kb_config=kb_config,
                    )
                    logits = out["logits"]

                    if a_step == 0 and step % 10 == 0:
                        batch_index = 0
                        max_logits = logits.argmax(axis=2)
                        decoded_pred = self.tokenizer.decode(max_logits[batch_index, :-1])
                        sel_labels = labels[batch_index, :]
                        sel_labels = sel_labels[sel_labels >= 0]
                        decoded_gt = self.tokenizer.decode(sel_labels)
                        self.logger.info(f"KB SHAPE: {kb_embedding[0].shape}")
                        self.logger.info(f"GT: {decoded_gt}")
                        self.logger.info(f"PRED: {decoded_pred}")
                        wandb.log({"kbsize": kb_embedding[0].shape[1]})

                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    weights = (shift_labels > 0).sum(-1, keepdim=True).expand(-1, shift_labels.shape[1]).contiguous()

                    model_config = (
                        self.model.config
                        if not isinstance(self.model, DistributedDataParallel)
                        else self.model.module.config
                    )
                    shift_logits = shift_logits.view(-1, model_config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    weights = weights.view(-1)
                    shift_labels = shift_labels.to(shift_logits.device)

                    if not self.config.use_certainty_loss:
                        loss = (loss_fct(shift_logits, shift_labels) * weights.max() / weights).mean()
                        self.accelerator.backward(loss)
                        losses.append(loss.item())
                    else:
                        ce_loss = (loss_fct(shift_logits, shift_labels) * weights.max() / weights).mean()
                        sce_loss = torch.logsumexp(shift_logits, dim=-1) - shift_logits.mean(dim=-1)
                        sce_loss = sce_loss.mean()
                        loss = 0.8 * ce_loss + 0.2 * sce_loss
                        self.accelerator.backward(loss)
                        losses.append(loss.item())

                self.optim.step()
                if self.config.use_lr_decay:
                    self.scheduler.step()

                if losses:
                    local_loss = torch.tensor(np.mean(losses), device=self.device)
                else:
                    local_loss = torch.tensor(0.0, device=self.device)

                all_losses = self.accelerator.gather(local_loss)
                valid_losses = all_losses[all_losses > 0]
                avg_loss = valid_losses.mean().item() if len(valid_losses) > 0 else 0.0

                if self.accelerator.is_main_process:
                    self.logger.info(f"step: {step}, loss: {avg_loss}")
                    wandb.log({'train_loss': np.mean(losses)})
                    train_losses.append(avg_loss)
                    pbar.update(task, advance=1, loss=avg_loss)

                if ((step % save_period) == 0 and (step != start_step)) or step == self.config.total_steps - 1:
                    self._save_checkpoint(step, kb_config)

    def _create_progress_bar(self) -> Progress:
        """Create progress bar for training."""
        columns = [
            SpinnerColumn(spinner_name="dots", style="cyan"),
            TextColumn("[bold blue]{task.description}", justify="right"),
            BarColumn(bar_width=None, style="cyan", complete_style="bold cyan"),
            TaskProgressColumn(),
            TextColumn("[bold yellow]Loss: {task.fields[loss]:.4f}", justify="right"),
            TimeRemainingColumn(),
        ]
        return Progress(*columns, console=console, expand=True, disable=not self.accelerator.is_main_process)

    def _save_checkpoint(self, step: int, kb_config: Union[KBLaMConfig, AtlasKVConfig]):
        """Save model checkpoint."""
        try:
            torch.cuda.empty_cache()
            self.accelerator.wait_for_everyone()

            if self.accelerator.is_main_process:
                self.logger.info("Saving checkpoint...")
                model_ckpt_name = self.output_dir / f"{self.config.train_dataset}_step_{step}"
                model_ckpt_name.mkdir(parents=True, exist_ok=True)

                encoder_dir = self.output_dir / f"{self.config.train_dataset}_step_{step}_encoder"
                encoder_dir.mkdir(parents=True, exist_ok=True)

                unwrapped_model = self.accelerator.unwrap_model(self.model)
                unwrapped_model.save_pretrained(
                    model_ckpt_name,
                    is_main_process=self.accelerator.is_main_process,
                    save_function=self.accelerator.save,
                )

                encoder_ckpt_name = encoder_dir / "encoder.pt"
                torch.save(self.retriever.encoder.state_dict(), encoder_ckpt_name)

                config_path = model_ckpt_name / "kb_config_explicit.json"
                with open(config_path, 'w') as f:
                    f.write(kb_config.to_json_string())

        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
            raise e


def create_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    """Create training configuration from command line arguments."""
    return TrainingConfig(
        seed=args.seed,
        train_dataset=args.train_dataset,
        N=args.N,
        B=args.B,
        lr=args.lr,
        total_steps=args.total_steps,
        encoder_spec=args.encoder_spec,
        key_embd_src=args.key_embd_src,
        gradient_accm_step=args.gradient_accm_step,
        kb_token_layer_frequency=args.kb_token_layer_frequency,
        llm_type=args.llm_type,
        hf_model_spec=args.hf_model_spec,
        max_seq_len=args.max_seq_len,
        kb_size=args.kb_size,
        dynamic_kb_size=args.dynamic_kb_size,
        outlier_num=args.outlier_num,
        multi_entities=args.multi_entities,
        sep_query_head=args.sep_query_head,
        use_data_aug=args.use_data_aug,
        use_lr_decay=args.use_lr_decay,
        use_cached_embd=args.use_cached_embd,
        use_certainty_loss=args.use_certainty_loss,
        use_kg=args.use_kg,
        use_hierarchial_kv=args.use_hierarchial_kv,
        use_extended_qa=args.use_extended_qa,
        duplicate_true_kb=args.duplicate_true_kb,
        length_invariance=args.length_invariance,
        projector_type=args.projector_type,
        verbose=args.verbose,
        log_to_file=args.log_to_file,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train_dataset", type=str, default="synthetic_data_QA")
    parser.add_argument("--N", type=int, default=120000, help="Size of training set")
    parser.add_argument("--B", type=int, default=10, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--sep_query_head", action="store_true", help="Train a separate query head")
    parser.add_argument("--use_oai_embd", action="store_true", help="Use OpenAI embedding")
    parser.add_argument("--use_cached_embd", action="store_true", help="Use pre-computed embeddings")
    parser.add_argument("--total_steps", type=int, default=20000, help="Total steps")
    parser.add_argument("--encoder_spec", type=str, default="OAI")
    parser.add_argument("--key_embd_src", type=str, default="key", choices=["key", "answer", "questions", None])
    parser.add_argument("--use_data_aug", action="store_true", help="Randomly pick templates")
    parser.add_argument("--use_lr_decay", action="store_true")
    parser.add_argument("--dataset_dir", type=str, default="your_dataset_dir")
    parser.add_argument("--model_dir_to_resume", type=str, default=None)
    parser.add_argument("--hf_model_spec", type=str, default="unsloth/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--model_save_dir", type=str, default="output")
    parser.add_argument("--kb_size", type=int, default=None)
    parser.add_argument("--dynamic_kb_size", nargs=2, type=int, default=None)
    parser.add_argument("--duplicate_true_kb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--length_invariance", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--outlier_num", type=int, default=1)
    parser.add_argument("--multi_entities", type=int, default=None)
    parser.add_argument("--use_extended_qa", action="store_true")
    parser.add_argument("--kb_token_layer_frequency", type=int, default=3)
    parser.add_argument("--gradient_accm_step", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log_to_file", action="store_true")
    parser.add_argument("--llm_type", type=str, default="llama3", choices=["llama3", "phi3"])
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--use_certainty_loss", action="store_true")
    parser.add_argument("--use_kg", action="store_true")
    parser.add_argument("--use_hierarchial_kv", action="store_true")
    parser.add_argument("--projector_type", type=str, default="linear", choices=["linear", "mlp"])
    return parser


def main():
    """Main training function."""
    logger = logging.getLogger("training")
    args = build_parser().parse_args()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    config = create_config_from_args(args)
    print(vars(args))

    set_llama_attention_classes(args.use_kg)

    if args.kb_size is not None and args.dynamic_kb_size is not None:
        raise ValueError("Can't specify kb_size and dynamic_kb_size. Use only one")

    kb_size = args.kb_size if args.kb_size is not None else args.dynamic_kb_size

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pathlib.Path(args.model_save_dir).mkdir(parents=True, exist_ok=True)

    if Accelerator().is_main_process:
        wandb.init(
            project="kb-llm",
            config={
                "learning_rate": config.lr,
                'sep_query_head': config.sep_query_head,
                'kb_size': kb_size,
                'length_invariance': config.length_invariance,
                'dataset': config.train_dataset,
                'outlier_num': config.outlier_num,
                'multi_entities': config.multi_entities,
                'use_extended_qa': config.use_extended_qa,
                'kb_token_layer_frequency': config.kb_token_layer_frequency,
                'gradient_accm_step': config.gradient_accm_step,
                "encoder_spec": config.encoder_spec,
                "max_seq_len": config.max_seq_len,
                "use_kg": config.use_kg,
                "use_hierarchial_kv": config.use_hierarchial_kv,
            },
        )

    torch.cuda.empty_cache()

    if args.log_to_file:
        formatter = logging.Formatter(LOGFORMAT)
        f_handler = logging.FileHandler(args.model_save_dir / "log.txt")
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)

    logger.info(f"Running on {device}")
    logger.info("🚨 Started training 🚨")
    logger.info(f"💽 Saving to {args.model_save_dir}💽")

    if args.sep_query_head:
        os.environ["SEP_QUERY_HEAD"] = "TRUE"
        logger.info("Having separate query head for KB!")

    if args.length_invariance:
        os.environ["LENGTH_INVARIANCE"] = "TRUE"
        logger.info("Having length invariance!")

    os.environ["SCALE_FACTOR"] = ""

    # Load dataset
    dataset = DataProcessor.load_dataset(args.dataset_dir, args.train_dataset, args.use_extended_qa)
    training_set = dataset[:args.N]

    # Setup model
    model_spec = args.model_dir_to_resume if args.model_dir_to_resume else args.hf_model_spec
    resumed_step = 0 if not args.model_dir_to_resume else int(args.model_dir_to_resume.split("_")[-1])

    if model_spec is None:
        raise ValueError("Either supply model_dir_to_resume or hf_model_spec")

    if args.hf_token is None and args.llm_type == "llama3":
        raise ValueError("Please supply HuggingFace token when loading Llama weights")

    tokenizer = ModelFactory.create_tokenizer(args.hf_model_spec, args.hf_token, args.llm_type)
    model = ModelFactory.create_model(model_spec, args.llm_type, args.use_kg, device, args.hf_token)

    logger.info(model.config)

    # Setup encoder
    encoder = ModelFactory.create_encoder(
        args.encoder_spec, model.config, args.kb_token_layer_frequency, args.projector_type, device
    )

    if args.model_dir_to_resume:
        encoder.load_state_dict(torch.load(os.path.join(args.model_dir_to_resume, "encoder.pt")))
        kb_config = KBLaMConfig.from_pretrained(os.path.join(args.model_dir_to_resume, "kb_config.json"))
    else:
        if config.use_kg:
            kb_config = AtlasKVConfig(
                sep_query_head=True,
                kb_layer_frequency=args.kb_token_layer_frequency,
                root_top_k_kb=500,
                inter_top_k_kb=500,
                leaf_top_k_kb=100000,
                use_hierarchial_kv=args.use_hierarchial_kv,
            )
        else:
            kb_config = KBLaMConfig(
                sep_query_head=args.sep_query_head,
                kb_layer_frequency=args.kb_token_layer_frequency,
                use_kg=False,
                use_hierarchial_kv=args.use_hierarchial_kv,
            )

    encoder.train()

    # Setup retriever
    if args.use_cached_embd:
        if args.use_hierarchial_kv:
            leaf_keys, inter_keys, root_keys, values, root_c2id, inter_c2id, root_id2c, inter_id2c = DataProcessor.load_cached_hierarchical_embeddings(
                args.encoder_spec, args.dataset_dir, args.train_dataset, args.key_embd_src
            )
            retriever = DataRetriever(
                encoder, dataset, None, None, leaf_keys, inter_keys, root_keys, values,
                root_c2id, inter_c2id, root_id2c, inter_id2c
            )
        else:
            key_embds, value_embds = DataProcessor.load_cached_embeddings(
                args.encoder_spec, args.dataset_dir, args.train_dataset, args.key_embd_src
            )
            retriever = DataRetriever(encoder, training_set, key_embds, value_embds)
    else:
        retriever = DataRetriever(encoder, training_set)

    logger.info("Model ready 🚀")

    # Start training
    trainer = ModelTrainer(model, retriever, tokenizer, config, device, args.model_save_dir)
    
    def _get_parameter_count(encoder):
        return sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    logger.info(f"Number of trainable parameters: {_get_parameter_count(encoder):,}")

    trainer.train(
        training_set,
        args.B,
        args.gradient_accm_step,
        args.outlier_num,
        use_data_aug=args.use_data_aug,
        multi_entities=args.multi_entities,
        use_extended_qa=args.use_extended_qa,
        save_period=1000,
        resumed_step=resumed_step,
        kb_config=kb_config,
    )


if __name__ == "__main__":
    main()
