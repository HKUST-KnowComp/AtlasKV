from typing import Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer

from atlaskv.models.kblam_processor import EncoderArgs, KBLaMProcessor
from atlaskv.models.llama3_model import KblamLlamaForCausalLM


class ModelLoader:
    """Handles model and processor loading operations."""
    
    @staticmethod
    def load_kblam_model(model_path: str) -> KblamLlamaForCausalLM:
        """Load KBLaM model from path."""
        return KblamLlamaForCausalLM.from_pretrained(model_path).bfloat16()

    @staticmethod
    def load_tokenizer(model_path: str) -> AutoTokenizer:
        """Load tokenizer from model path."""
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    @staticmethod
    def create_encoder_args(encoder_name: str, model_config, kb_layer_frequency: int, encoder_dir: str) -> EncoderArgs:
        """Create encoder arguments from model configuration."""
        return EncoderArgs(
            encoder_name=encoder_name,
            hidden_size=model_config.hidden_size,
            num_hidden_layers=model_config.num_hidden_layers,
            kb_layer_frequency=kb_layer_frequency,
            encoder_dir=encoder_dir,
        )

    @classmethod
    def load_model_and_processor(
        cls, model_path: str, encoder_name: str, kb_layer_frequency: int, encoder_dir: str
    ) -> Tuple[AutoModelForCausalLM, KBLaMProcessor]:
        """Load model and processor with given configuration."""
        # Load model and tokenizer
        model = cls.load_kblam_model(model_path)
        tokenizer = cls.load_tokenizer(model_path)
        
        # Create encoder arguments
        args = cls.create_encoder_args(encoder_name, model.config, kb_layer_frequency, encoder_dir)
        
        # Create processor
        processor = KBLaMProcessor(tokenizer, args)
        
        return model, processor


# Backward compatibility function
def load_model_and_processor(
    model_path: str, encoder_name: str, kb_layer_frequency: int, encoder_dir: str
) -> Tuple[AutoModelForCausalLM, KBLaMProcessor]:
    """Load model and processor (backward compatibility)."""
    return ModelLoader.load_model_and_processor(model_path, encoder_name, kb_layer_frequency, encoder_dir)
