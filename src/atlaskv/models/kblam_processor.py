from dataclasses import dataclass
from typing import Union

import torch
from transformers import AutoTokenizer, BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from atlaskv.kb_encoder import KBEncoder


@dataclass
class EncoderArgs:
    """Configuration arguments for KBLaM encoder."""
    encoder_name: str
    hidden_size: int
    num_hidden_layers: int
    kb_layer_frequency: int
    encoder_dir: str
    projector_type: str
    endpoint_url: str


class EncoderLoader:
    """Handles KBLaM encoder loading and configuration."""
    
    @staticmethod
    def create_encoder(args: EncoderArgs) -> KBEncoder:
        """Create and configure KBLaM encoder."""
        encoder = KBEncoder(
            encoder_name=args.encoder_name,
            projector_type=args.projector_type,
            endpoint_url=args.endpoint_url,
            out_dim=args.hidden_size * (args.num_hidden_layers // args.kb_layer_frequency + 1),
            frozen_base_model=True,
            projector_kwargs={"mlp_depth": 1, "mlp_hidden_dim": 512},
            get_oai_embd_online=False,
        )
        
        encoder.load_state_dict(torch.load(args.encoder_dir))
        return encoder


class TextProcessor:
    """Handles text processing and formatting."""
    
    @staticmethod
    def format_llama_text(text: str) -> str:
        """Format text for Llama model."""
        return (
            "<|start_header_id|>user<|end_header_id|> "
            + text
            + "<|eot_id|>"
            + "<|start_header_id|>assistant<|end_header_id|>"
        )


class KnowledgeBaseProcessor:
    """Handles knowledge base processing."""
    
    def __init__(self, kb_encoder: KBEncoder):
        self.kb_encoder = kb_encoder

    def process_knowledge_base(self, knowledge_base) -> Union[list, None]:
        """Process knowledge base based on input type."""
        if not knowledge_base:
            return None
        
        if not isinstance(knowledge_base, list):
            return knowledge_base
        
        # Process tensor embeddings
        if isinstance(knowledge_base[0][0], torch.Tensor):
            return self.kb_encoder.encode_base_embeddings(knowledge_base)
        
        # Process string inputs
        if isinstance(knowledge_base[0][0], str):
            return self.kb_encoder.encode(knowledge_base)
        
        return knowledge_base


class TokenizerWrapper:
    """Handles tokenizer operations and device management."""
    
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._setup_tokenizer()

    def _setup_tokenizer(self):
        """Setup tokenizer with appropriate pad token."""
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_text(self, text: str) -> dict:
        """Tokenize text and move to device."""
        return self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)

    def batch_decode(self, *args, **kwargs):
        """Forward to tokenizer batch decode."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Forward to tokenizer decode."""
        return self.tokenizer.decode(*args, **kwargs)


class KBLaMProcessor(ProcessorMixin):
    """Alternative KBLaM processor with modular design."""
    
    feature_extractor_class = "AutoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, tokenizer: AutoTokenizer, args: EncoderArgs, **kwargs):
        # Initialize components
        self.kb_encoder = EncoderLoader.create_encoder(args)
        self.tokenizer_wrapper = TokenizerWrapper(tokenizer)
        self.text_processor = TextProcessor()
        self.kb_processor = KnowledgeBaseProcessor(self.kb_encoder)
        
        # Store references for compatibility
        self.tokenizer = tokenizer
        self.device = self.tokenizer_wrapper.device
        
        super().__init__(self.kb_encoder, self.tokenizer)

    def __call__(
        self,
        knowledge_base: list[tuple[torch.Tensor]] | list[tuple[str]] = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
    ) -> BatchFeature:
        """Process knowledge base and text inputs."""
        # Process knowledge base
        processed_kb = self.kb_processor.process_knowledge_base(knowledge_base)
        
        # Process text
        formatted_text = self.text_processor.format_llama_text(text)
        text_inputs = self.tokenizer_wrapper.tokenize_text(formatted_text)
        
        return BatchFeature(data={**text_inputs, "kb_kvs": processed_kb})

    def batch_decode(self, *args, **kwargs):
        """Forward to tokenizer batch decode."""
        return self.tokenizer_wrapper.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Forward to tokenizer decode."""
        return self.tokenizer_wrapper.decode(*args, **kwargs)
