import argparse
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel
from torch.optim.optimizer import ParamsT


class TensorUtils:
    """Handles tensor configuration and operations."""
    
    @staticmethod
    def get_tensor_config(tensor: torch.Tensor) -> Dict[str, any]:
        """Get tensor configuration dictionary."""
        return {"dtype": tensor.dtype, "layout": tensor.layout, "device": tensor.device}


class EmbeddingProcessor:
    """Handles embedding preprocessing and processing."""
    
    @staticmethod
    def preprocess_embeddings(emb1: List, emb2: List, kb_mask_val: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Preprocess embeddings by padding and concatenating.
        
        Args:
            emb1: List of KB embeddings
            emb2: List of query embeddings  
            kb_mask_val: Attention mask value for KB part
            
        Returns:
            Tuple of (joint_embeddings, attention_masks, position_ids, kb_masks)
        """
        assert isinstance(emb1, list)
        assert isinstance(emb2, list)
        assert len(emb1) == len(emb2)
        
        max_length = max([e1.shape[0] + e2.shape[0] for e1, e2 in zip(emb1, emb2)])
        joint_embs = []
        attention_masks = []
        position_ids = []
        kb_masks = []
        
        for e1, e2 in zip(emb1, emb2):
            tensor_config = TensorUtils.get_tensor_config(e1)
            pad_size = max_length - e1.shape[0] - e2.shape[0]
            padding = torch.zeros((pad_size, e1.shape[1]), **tensor_config)
            joint_embs.append(torch.concat([padding, e1, e2]))
            
            # Create attention mask
            attention_mask = torch.cat([
                torch.zeros(pad_size, **tensor_config),
                torch.zeros(e1.shape[0], **tensor_config) + kb_mask_val,
                torch.ones(e2.shape[0], **tensor_config),
            ])
            attention_masks.append(attention_mask)
            
            # Create position IDs
            position_id = torch.cat([
                torch.zeros(max_length - e2.shape[0], **tensor_config) - 1,
                torch.arange(1, e2.shape[0] + 1, **tensor_config) - 1,
            ])
            position_ids.append(position_id)
            
            # Create KB mask
            kb_mask = torch.cat([
                torch.zeros(pad_size, **tensor_config),
                torch.ones(e1.shape[0], **tensor_config),
                torch.zeros(e2.shape[0], **tensor_config),
            ])
            kb_masks.append(kb_mask)
        
        return (
            torch.stack(joint_embs),
            torch.stack(attention_masks),
            torch.stack(position_ids),
            torch.stack(kb_masks),
        )


class KBEncoder:
    """Handles knowledge base encoding operations."""
    
    @staticmethod
    def encode_kb_embeddings(kb_encoder, kb_dict=None, precomputed_base_embd=None):
        """Encode KB embeddings from dictionary or precomputed embeddings."""
        if isinstance(kb_encoder, DistributedDataParallel):
            kb_encoder = kb_encoder.module
        
        key_embds, value_embds = [], []
        if precomputed_base_embd is not None:
            if not torch.is_grad_enabled():
                print(f"\033[94m[Encoding Precomputed Base Embeddings]\033[0m")
            key_embds = kb_encoder.encode_key(base_emb=precomputed_base_embd[0])
            value_embds = kb_encoder.encode_val(base_emb=precomputed_base_embd[1])
        else:
            for entity in kb_dict:
                key_embds.append(kb_encoder.encode_key(S=entity["key_string"]))
                value_embds.append(kb_encoder.encode_val(S=entity["description"]))
        return (key_embds, value_embds)

    @staticmethod
    def encode_kg_embeddings(kb_encoder, kb_dict=None, precomputed_base_embd=None, mode: str = "train"):
        """Encode hierarchical knowledge graph embeddings."""
        if isinstance(kb_encoder, DistributedDataParallel):
            kb_encoder = kb_encoder.module
        
        if precomputed_base_embd is not None:
            root_key_base_embds, inter_key_base_embds, leaf_key_base_embds, value_base_embds = precomputed_base_embd
            if not torch.is_grad_enabled():
                print(f"\033[94m[Encoding Precomputed Base Embeddings]\033[0m")
            
            if mode == "train":
                root_key_embds = kb_encoder.encode_key(base_emb=root_key_base_embds)
                inter_key_embds = kb_encoder.encode_key(base_emb=inter_key_base_embds)
                leaf_key_embds = kb_encoder.encode_key(base_emb=leaf_key_base_embds)
                value_embds = kb_encoder.encode_val(base_emb=value_base_embds)
            elif mode == "eval":
                root_key_embds = kb_encoder.encode_key(base_emb=root_key_base_embds)
                inter_key_embds = kb_encoder.encode_key_cpu(base_emb=inter_key_base_embds)
                leaf_key_embds = kb_encoder.encode_key_cpu(base_emb=leaf_key_base_embds)
                value_embds = kb_encoder.encode_val_cpu(base_emb=value_base_embds)
        else:
            raise NotImplementedError("Precomputed base embeddings are not supported for AtlasKV now.")
        
        return (root_key_embds, inter_key_embds, leaf_key_embds, value_embds)


class EmbeddingRetriever:
    """Handles embedding retrieval for training."""
    
    @staticmethod
    def get_kb_embeddings(kb_encoder, indices: List[int], kb_dict: Dict = None, precomputed_embd: Tuple[torch.Tensor] = None) -> Tuple[torch.Tensor]:
        """Get KB embeddings for given indices."""
        if precomputed_embd:
            key_embds, value_embds = precomputed_embd
            train_set_key, train_set_val = KBEncoder.encode_kb_embeddings(
                kb_encoder,
                precomputed_base_embd=np.stack([key_embds[indices], value_embds[indices]]),
            )
        elif kb_dict:
            if len(indices.shape) == 2:
                train_set_key, train_set_val = [], []
                for indices_row in indices.T:
                    _train_set_key, _train_set_val = KBEncoder.encode_kb_embeddings(
                        kb_encoder, kb_dict=[kb_dict[i] for i in indices_row]
                    )
                    train_set_key.append(_train_set_key)
                    train_set_val.append(_train_set_val)
                train_set_key = torch.stack(train_set_key, 1)
                train_set_val = torch.stack(train_set_val, 1)
            elif len(indices.shape) == 1:
                train_set_key, train_set_val = KBEncoder.encode_kb_embeddings(
                    kb_encoder, kb_dict=[kb_dict[i] for i in indices]
                )
        return train_set_key, train_set_val

    @staticmethod
    def get_hierarchical_kg_embeddings(
        kb_encoder, indices: List[int], root_id2c: Dict = None, inter_id2c: Dict = None,
        root_c2id: Dict = None, inter_c2id: Dict = None, kb_dict: Dict = None,
        mode: str = "train", precomputed_embd: Tuple[torch.Tensor] = None
    ) -> Tuple[torch.Tensor]:
        """Get hierarchical KG embeddings for given indices."""
        if precomputed_embd:
            leaf_key_embds, inter_key_embds, root_key_embds, value_embds = precomputed_embd
            
            if torch.is_grad_enabled():
                leaf_key_indices = np.array(indices)
                if leaf_key_indices.ndim == 1:
                    leaf_key_indices = np.array([leaf_key_indices])
                
                inter_key_indices = np.zeros(leaf_key_indices.shape, dtype=int)
                for i, i_leaf_key_indices in enumerate(leaf_key_indices):
                    i_inter_key_indices = np.array(list([inter_id2c[str(int(i))] for i in i_leaf_key_indices]))
                    inter_key_indices[i] = i_inter_key_indices
                
                root_key_indices = np.zeros(inter_key_indices.shape, dtype=int)
                for i, i_inter_key_indices in enumerate(inter_key_indices):
                    i_root_key_indices = np.array(list([root_id2c[str(int(i))] for i in i_inter_key_indices]))
                    root_key_indices[i] = i_root_key_indices
            else:
                leaf_key_indices = indices
                inter_key_indices = list(set([inter_id2c[str(int(i))] for i in leaf_key_indices]))
                root_key_indices = list(set([root_id2c[str(int(i))] for i in inter_key_indices]))
            
            train_set_root_key, train_set_inter_key_base_embd, train_set_leaf_key_base_embd, train_set_val_base_embd = KBEncoder.encode_kg_embeddings(
                kb_encoder,
                precomputed_base_embd=(root_key_embds[root_key_indices], inter_key_embds[inter_key_indices], leaf_key_embds[leaf_key_indices], value_embds[leaf_key_indices]),
                mode=mode,
            )
        elif kb_dict:
            if len(indices.shape) == 2:
                train_set_key, train_set_val = [], []
                for indices_row in indices.T:
                    _train_set_key, _train_set_val = KBEncoder.encode_kg_embeddings(
                        kb_encoder, kb_dict=[kb_dict[i] for i in indices_row], mode=mode
                    )
                    train_set_key.append(_train_set_key)
                    train_set_val.append(_train_set_val)
                train_set_key = torch.stack(train_set_key, 1)
                train_set_val = torch.stack(train_set_val, 1)
            elif len(indices.shape) == 1:
                train_set_key, train_set_val = KBEncoder.encode_kg_embeddings(
                    kb_encoder, kb_dict=[kb_dict[i] for i in indices], mode=mode
                )
        
        return (train_set_root_key, train_set_inter_key_base_embd, train_set_leaf_key_base_embd, 
                train_set_val_base_embd, torch.tensor(root_key_indices), torch.tensor(inter_key_indices), 
                torch.tensor(leaf_key_indices), root_c2id, inter_c2id)


class LossCalculator:
    """Handles loss calculation operations."""
    
    @staticmethod
    def compute_weighted_nll(model, input_ids, attention_mask, labels, kb=None):
        """Compute weighted negative log-likelihood loss."""
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            kb_kv=kb,
            output_attentions=True,
        )
        logits = out["logits"]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        weights = (shift_labels > 0).sum(-1, keepdim=True).expand(-1, shift_labels.shape[1]).contiguous()
        shift_logits = shift_logits.view(-1, model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        weights = weights.view(-1)
        loss_fct = CrossEntropyLoss(reduction="none")
        shift_labels = shift_labels.to(shift_logits.device)
        loss = (loss_fct(shift_logits, shift_labels) * weights.max() / weights).mean()
        return loss

    @staticmethod
    def compute_perplexity_gain(model, kb, input_ids, attention_mask, labels):
        """Compute perplexity gain with and without KB."""
        with torch.autograd.no_grad():
            unconditioned_nll = LossCalculator.compute_weighted_nll(model, input_ids, attention_mask, labels, kb=None)
            conditioned_nll = LossCalculator.compute_weighted_nll(model, input_ids, attention_mask, labels, kb)
        return unconditioned_nll, conditioned_nll


class Scheduler:
    """Handles scheduling operations."""
    
    @staticmethod
    def context_set_size_scheduler(curr_step: int, kb_size: Union[List[int], int, str]) -> int:
        """Determine KB size for current training step."""
        dynamic_range = (10, 200)
        if kb_size == "dynamic":
            return np.random.randint(dynamic_range[0], dynamic_range[1])
        
        if isinstance(kb_size, list):
            return np.random.randint(kb_size[0], kb_size[1])
        
        increase_kb_size_every = 100
        if not kb_size:
            round_num = (curr_step) // increase_kb_size_every
            return 4 * (round_num + 1)
        
        return kb_size

    @staticmethod
    def setup_scheduler_and_optimizer(model_parameters: ParamsT, lr: float, max_iter: int) -> Tuple:
        """Setup learning rate scheduler and optimizer."""
        optim = torch.optim.AdamW(model_parameters, lr=lr, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_iter, eta_min=lr * 0.01)
        return scheduler, optim


class PrefixGenerator:
    """Handles prefix string generation for experiments."""
    
    @staticmethod
    def get_prefix_string(args: argparse.Namespace) -> str:
        """Generate prefix string from arguments."""
        kb_size = args.kb_size
        if kb_size == -1:
            kb_size = None
        elif kb_size == 0:
            kb_size = "dynamic"
        
        prefix_string = f"stage1_lr_{args.lr}"
        if args.kb_token_layer_frequency is not None:
            prefix_string += f"KBTokenLayerFreq{args.kb_token_layer_frequency}"
        if args.use_extended_qa:
            prefix_string += "UseExtendedQA"
        if args.multi_entities is not None:
            prefix_string += f"MultiEntities{args.multi_entities}"
        if args.outlier_num > 0:
            prefix_string += f"UseOutlier{args.outlier_num}"
        if args.length_invariance:
            prefix_string += "LengthInvariant"
        if kb_size is not None:
            prefix_string += f"KBSize{kb_size}"
        if args.sep_query_head:
            prefix_string += "SepQueryHead"
        if args.use_data_aug:
            prefix_string += "UseDataAug"
        return prefix_string


# Backward compatibility functions
def get_tensor_config(x: torch.Tensor) -> Dict[str, any]:
    """Get tensor config (backward compatibility)."""
    return TensorUtils.get_tensor_config(x)


def preprocess_embds(emb1: List, emb2: List, kb_mask_val: int = 1):
    """Preprocess embeddings (backward compatibility)."""
    return EmbeddingProcessor.preprocess_embeddings(emb1, emb2, kb_mask_val)


def kb_to_embd(kb_encoder, kb_dict=None, precomputed_base_embd=None):
    """Encode KB embeddings (backward compatibility)."""
    return KBEncoder.encode_kb_embeddings(kb_encoder, kb_dict, precomputed_base_embd)


def kg_to_embd(kb_encoder, kb_dict=None, precomputed_base_embd=None, mode: str = "train"):
    """Encode KG embeddings (backward compatibility)."""
    return KBEncoder.encode_kg_embeddings(kb_encoder, kb_dict, precomputed_base_embd, mode)


def get_kb_embd(kb_encoder, indices: List[int], kb_dict: Dict = None, precomputed_embd: Tuple[torch.Tensor] = None):
    """Get KB embeddings (backward compatibility)."""
    return EmbeddingRetriever.get_kb_embeddings(kb_encoder, indices, kb_dict, precomputed_embd)


def get_hierarchical_kg_embd(kb_encoder, indices: List[int], root_id2c: Dict = None, inter_id2c: Dict = None,
                            root_c2id: Dict = None, inter_c2id: Dict = None, kb_dict: Dict = None,
                            mode: str = "train", precomputed_embd: Tuple[torch.Tensor] = None):
    """Get hierarchical KG embeddings (backward compatibility)."""
    return EmbeddingRetriever.get_hierarchical_kg_embeddings(
        kb_encoder, indices, root_id2c, inter_id2c, root_c2id, inter_c2id, kb_dict, mode, precomputed_embd
    )


def weighted_nll(model, input_ids, attention_mask, labels, kb=None):
    """Compute weighted NLL (backward compatibility)."""
    return LossCalculator.compute_weighted_nll(model, input_ids, attention_mask, labels, kb)


def compute_perplexity_gain(model, kb, input_ids, attention_mask, labels):
    """Compute perplexity gain (backward compatibility)."""
    return LossCalculator.compute_perplexity_gain(model, kb, input_ids, attention_mask, labels)


def context_set_size_scheduler(curr_step: int, kb_size: Union[List[int], int, str]) -> int:
    """Context set size scheduler (backward compatibility)."""
    return Scheduler.context_set_size_scheduler(curr_step, kb_size)


def get_prefix_str(args: argparse.Namespace) -> str:
    """Get prefix string (backward compatibility)."""
    return PrefixGenerator.get_prefix_string(args)


def setup_scheduler_and_optimizer(model_parapmeters: ParamsT, lr: float, max_iter: int) -> Tuple:
    """Setup scheduler and optimizer (backward compatibility)."""
    return Scheduler.setup_scheduler_and_optimizer(model_parapmeters, lr, max_iter)
