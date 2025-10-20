import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from sentence_transformers import SentenceTransformer
from transformers import FeatureExtractionMixin
from tqdm import tqdm

from .gpt_session import GPT


@dataclass
class EncoderConfig:
    """Configuration for knowledge base encoder."""
    encoder_name: str = ""
    projector_type: str = "linear"
    out_dim: int = 768
    endpoint_url: str = ""
    endpoint_api_key: str = ""
    projector_kwargs: Dict = None
    frozen_base_model: bool = True
    device: Union[str, torch.device] = "cuda"
    get_oai_embd_online: bool = False
    encoding_batch_size: int = 512

    def __post_init__(self):
        if self.projector_kwargs is None:
            self.projector_kwargs = {}


class ProjectorFactory:
    """Factory for creating projector modules."""
    
    @staticmethod
    def create_identity_map() -> nn.Module:
        """Create identity mapping module."""
        class IdentityMapper(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, *args, **kwargs):
                return x
        return IdentityMapper()

    @staticmethod
    def create_linear_projector(in_dim: int, out_dim: int) -> nn.Module:
        """Create linear projector module."""
        return nn.Linear(in_dim, out_dim)

    @staticmethod
    def create_mlp_projector(in_dim: int, out_dim: int, config: Dict) -> nn.Module:
        """Create MLP projector module."""
        mlp_depth = config.get("mlp_depth", 2)
        mlp_hidden_dim = config.get("mlp_hidden_dim", 1024)
        
        modules = [nn.Linear(in_dim, mlp_hidden_dim)]
        for _ in range(mlp_depth):
            modules.append(nn.Linear(mlp_hidden_dim, mlp_hidden_dim))
            modules.append(nn.GELU())
        modules.append(nn.Linear(mlp_hidden_dim, out_dim))
        return nn.Sequential(*modules)

    @classmethod
    def build_projector(cls, projector_type: str, in_dim: int, out_dim: int, 
                       projector_kwargs: Dict) -> nn.Module:
        """Build projector based on type and configuration."""
        if projector_type == "identity":
            return cls.create_identity_map()
        elif projector_type == "linear":
            return cls.create_linear_projector(in_dim, out_dim)
        elif projector_type == "mlp":
            return cls.create_mlp_projector(in_dim, out_dim, projector_kwargs)
        else:
            raise NotImplementedError(f"Projector type {projector_type} not found")


class SpecialTokenManager:
    """Manages special tokens for knowledge base encoding."""
    
    SPECIAL_TOKENS = {
        "<KB_BEGIN>": 0,
        "<KB_END>": 1,
        "<KEY_SEP>": 2,
        "<VALUE_SEP>": 3,
        "<ENTITY_SEP>": 4,
        "<KV_SEP>": 5,
    }

    def __init__(self, out_dim: int, device: Union[str, torch.device]):
        self.out_dim = out_dim
        self.device = device
        self.embedding = nn.Embedding(len(self.SPECIAL_TOKENS), out_dim)

    def get_token_embedding(self, token_type: str) -> torch.Tensor:
        """Get embedding for special token."""
        idx = torch.tensor(self.SPECIAL_TOKENS[token_type]).to(self.embedding.weight.device)
        return self.embedding(idx).bfloat16()


class BaseModelManager:
    """Manages base model initialization and encoding."""
    
    def __init__(self, config: EncoderConfig):
        self.config = config
        self.device = config.device
        self._setup_base_model()

    def _setup_base_model(self):
        """Setup base model based on encoder specification."""
        if self.config.encoder_name in ["OAI", "BigOAI"]:
            self._setup_openai_model()
        else:
            self._setup_sentence_transformer()

    def _setup_openai_model(self):
        """Setup OpenAI embedding model."""
        is_big = "Big" in self.config.encoder_name
        
        if self.config.get_oai_embd_online:
            model_name = "text-embedding-3-large" if is_big else "text-embedding-ada-002"
            self.gpt_session = GPT(model_name, self.config.endpoint_url, self.config.endpoint_api_key)
            self.base_model_encode = lambda s: torch.tensor(
                self.gpt_session.generate_embedding(s)
            ).to(self.device)
        else:
            self.base_model_encode = None
        
        self.in_dim = 3072 if is_big else 1536

    def _setup_sentence_transformer(self):
        """Setup SentenceTransformer model."""
        self.base_model = SentenceTransformer(self.config.encoder_name)
        self.base_model_encode = lambda s: self.base_model.encode(s, convert_to_numpy=False)
        
        if self.config.frozen_base_model:
            self.base_model.eval()
            for param in self.base_model.parameters():
                param.requires_grad = False
        else:
            self.base_model.train()
        
        self.in_dim = self.base_model.get_sentence_embedding_dimension()

    def encode_text(self, text_input: Optional[str] = None, 
                   base_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode text or use provided base embedding."""
        if text_input is not None:
            return self.base_model_encode(text_input)
        elif base_embedding is not None:
            return torch.from_numpy(base_embedding).to(self.device)
        else:
            raise ValueError("Either text_input or base_embedding must be provided")


class BatchProcessor:
    """Handles batch processing for encoding operations."""
    
    def __init__(self, batch_size: int, device: Union[str, torch.device]):
        self.batch_size = batch_size
        self.device = device

    def process_batches(self, base_embedding: torch.Tensor, 
                       processor_func, use_tqdm: bool = True) -> torch.Tensor:
        """Process embeddings in batches."""
        num_samples = base_embedding.size(0)
        embeddings = []
        
        iterator = range(0, num_samples, self.batch_size)
        if use_tqdm and not torch.is_grad_enabled():
            iterator = tqdm(iterator)
        
        for i in iterator:
            batch = base_embedding[i:i + self.batch_size]
            processed_batch = processor_func(batch)
            embeddings.append(processed_batch.bfloat16())
        
        return torch.cat(embeddings, dim=0)

    def process_batches_cpu(self, base_embedding: torch.Tensor, 
                           processor_func) -> torch.Tensor:
        """Process embeddings in batches and move to CPU."""
        num_samples = base_embedding.size(0)
        embeddings = []
        
        for i in tqdm(range(0, num_samples, self.batch_size)):
            batch = base_embedding[i:i + self.batch_size]
            processed_batch = processor_func(batch)
            cpu_batch = processed_batch.to(dtype=torch.bfloat16, device='cpu')
            embeddings.append(cpu_batch)
        
        result = torch.cat(embeddings, dim=0)
        self._cleanup_gpu_memory()
        return result

    def _cleanup_gpu_memory(self):
        """Clean up GPU memory."""
        torch.cuda.empty_cache()


class KeyEncoder:
    """Handles key encoding operations."""
    
    def __init__(self, projector: nn.Module, layernorm: nn.Module, 
                 batch_processor: BatchProcessor):
        self.projector = projector
        self.layernorm = layernorm
        self.batch_processor = batch_processor

    def encode_keys(self, text_input: Optional[str] = None, 
                   base_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode keys using projector and layer normalization."""
        if text_input is not None:
            base_emb = self._get_base_embedding(text_input)
        elif base_embedding is not None:
            base_emb = torch.from_numpy(base_embedding).to(self.batch_processor.device)
        else:
            raise ValueError("Either text_input or base_embedding must be provided")
        
        def process_batch(batch):
            return self.layernorm(self.projector(batch))
        
        return self.batch_processor.process_batches(base_emb, process_batch)

    def encode_keys_cpu(self, text_input: Optional[str] = None, 
                       base_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode keys and move results to CPU."""
        if text_input is not None:
            base_emb = self._get_base_embedding(text_input)
        elif base_embedding is not None:
            base_emb = torch.from_numpy(base_embedding).to(self.batch_processor.device)
        else:
            raise ValueError("Either text_input or base_embedding must be provided")
        
        def process_batch(batch):
            return self.layernorm(self.projector(batch))
        
        return self.batch_processor.process_batches_cpu(base_emb, process_batch)

    def _get_base_embedding(self, text_input: str) -> torch.Tensor:
        """Get base embedding from text input."""
        # This would be implemented by the base model manager
        raise NotImplementedError("Base model encoding should be handled by BaseModelManager")


class ValueEncoder:
    """Handles value encoding operations."""
    
    def __init__(self, projector: nn.Module, batch_processor: BatchProcessor):
        self.projector = projector
        self.batch_processor = batch_processor

    def encode_values(self, text_input: Optional[str] = None, 
                     base_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode values using projector."""
        if text_input is not None:
            base_emb = self._get_base_embedding(text_input)
        elif base_embedding is not None:
            base_emb = torch.from_numpy(base_embedding).to(self.batch_processor.device)
        else:
            raise ValueError("Either text_input or base_embedding must be provided")
        
        def process_batch(batch):
            return self.projector(batch)
        
        return self.batch_processor.process_batches(base_emb, process_batch)

    def encode_values_cpu(self, text_input: Optional[str] = None, 
                         base_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode values and move results to CPU."""
        if text_input is not None:
            base_emb = self._get_base_embedding(text_input)
        elif base_embedding is not None:
            base_emb = torch.from_numpy(base_embedding).to(self.batch_processor.device)
        else:
            raise ValueError("Either text_input or base_embedding must be provided")
        
        def process_batch(batch):
            return self.projector(batch)
        
        return self.batch_processor.process_batches_cpu(base_emb, process_batch)

    def _get_base_embedding(self, text_input: str) -> torch.Tensor:
        """Get base embedding from text input."""
        # This would be implemented by the base model manager
        raise NotImplementedError("Base model encoding should be handled by BaseModelManager")


class KBEncoder(nn.Module, FeatureExtractionMixin):
    """Alternative knowledge base encoder with modular design."""
    
    def __init__(
        self,
        encoder_name: str,
        projector_type: str,
        out_dim: int,
        endpoint_url: str,
        endpoint_api_key: str,
        projector_kwargs: Dict = None,
        frozen_base_model: bool = True,
        device: Union[str, torch.device] = "cuda",
        get_oai_embd_online: bool = False,
        encoding_batch_size: int = 512,
    ):
        super().__init__()
        
        # Create configuration
        self.config = EncoderConfig(
            encoder_name=encoder_name,
            projector_type=projector_type,
            out_dim=out_dim,
            endpoint_url=endpoint_url,
            endpoint_api_key=endpoint_api_key,
            projector_kwargs=projector_kwargs or {},
            frozen_base_model=frozen_base_model,
            device=device,
            get_oai_embd_online=get_oai_embd_online,
            encoding_batch_size=encoding_batch_size,
        )
        
        # Initialize components
        self.base_model_manager = BaseModelManager(self.config)
        self.batch_processor = BatchProcessor(encoding_batch_size, device)
        self.special_token_manager = SpecialTokenManager(out_dim, device)
        
        # Create projectors
        self.key_projector = ProjectorFactory.build_projector(
            projector_type, self.base_model_manager.in_dim, out_dim, 
            self.config.projector_kwargs
        )
        self.value_projector = ProjectorFactory.build_projector(
            projector_type, self.base_model_manager.in_dim, out_dim, 
            self.config.projector_kwargs
        )
        
        # Create layer normalization for keys
        self.key_layernorm = nn.LayerNorm(out_dim, elementwise_affine=False, bias=False)
        
        # Initialize encoders
        self.key_encoder = KeyEncoder(self.key_projector, self.key_layernorm, self.batch_processor)
        self.value_encoder = ValueEncoder(self.value_projector, self.batch_processor)
        
        # Set device
        self.device = device
        self.to(self.device)

    def freeze_value_projector(self):
        """Freeze value projector parameters."""
        for param in self.value_projector.parameters():
            param.requires_grad = False

    def encode_key(self, S: Optional[str] = None, base_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode keys using the backbone model and adapter."""
        if S is not None:
            base_embedding = self.base_model_manager.base_model_encode(S)
        elif base_emb is not None:
            base_embedding = torch.from_numpy(base_emb).to(self.device)
        else:
            raise ValueError("Either S or base_emb must be provided")
        
        def process_batch(batch):
            return self.key_layernorm(self.key_projector(batch))
        
        return self.batch_processor.process_batches(base_embedding, process_batch)

    def encode_key_cpu(self, S: Optional[str] = None, base_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode keys and move results to CPU."""
        if S is not None:
            base_embedding = self.base_model_manager.base_model_encode(S)
        elif base_emb is not None:
            base_embedding = torch.from_numpy(base_emb).to(self.device)
        else:
            raise ValueError("Either S or base_emb must be provided")
        
        def process_batch(batch):
            return self.key_layernorm(self.key_projector(batch))
        
        return self.batch_processor.process_batches_cpu(base_embedding, process_batch)

    def encode_val(self, S: Optional[str] = None, base_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode values using the backbone model and adapter."""
        if S is not None:
            base_embedding = self.base_model_manager.base_model_encode(S)
        elif base_emb is not None:
            base_embedding = torch.from_numpy(base_emb).to(self.device)
        else:
            raise ValueError("Either S or base_emb must be provided")
        
        def process_batch(batch):
            return self.value_projector(batch)
        
        return self.batch_processor.process_batches(base_embedding, process_batch)

    def encode_val_cpu(self, S: Optional[str] = None, base_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode values and move results to CPU."""
        if S is not None:
            base_embedding = self.base_model_manager.base_model_encode(S)
        elif base_emb is not None:
            base_embedding = torch.from_numpy(base_emb).to(self.device)
        else:
            raise ValueError("Either S or base_emb must be provided")
        
        def process_batch(batch):
            return self.value_projector(batch)
        
        return self.batch_processor.process_batches_cpu(base_embedding, process_batch)

    def encode_key_value(self, key: str, value: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode key-value pairs."""
        key_embedding = self.encode_key(S=key)
        value_embedding = self.encode_val(S=value)
        return key_embedding, value_embedding

    def encode_key_value_embeddings(self, key_embd: torch.Tensor, value_embd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode key-value embeddings."""
        key_embedding = self.encode_key(base_emb=key_embd)
        value_embedding = self.encode_val(base_emb=value_embd)
        return key_embedding, value_embedding

    def encode_base_embeddings(self, kb: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode knowledge base from base embeddings."""
        key_embeddings, value_embeddings = [], []
        for key, value in zip(kb[0], kb[1]):
            key_emb, value_emb = self.encode_key_value_embeddings(key, value)
            key_embeddings.append(key_emb)
            value_embeddings.append(value_emb)
        return torch.stack(key_embeddings), torch.stack(value_embeddings)

    def encode(self, kb: List[Tuple[str, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode knowledge base from text pairs."""
        key_embeddings, value_embeddings = [], []
        for key, value in kb:
            key_emb, value_emb = self.encode_key_value(key, value)
            key_embeddings.append(key_emb)
            value_embeddings.append(value_emb)
        return torch.stack(key_embeddings), torch.stack(value_embeddings)

    def get_special_token_embd(self, token_type: str) -> torch.Tensor:
        """Get embedding for special token."""
        return self.special_token_manager.get_token_embedding(token_type)

    @property
    def kb_special_token(self) -> Dict[str, int]:
        """Get special token mapping."""
        return SpecialTokenManager.SPECIAL_TOKENS
