from typing import Optional, Union

import numpy as np
import torch
import transformers

from atlaskv.models.kblam_config import KBLaMConfig
from atlaskv.models.llama3_model import KblamLlamaForCausalLM, AtlaskvLlamaForCausalLM
from atlaskv.models.phi3_model import KBLaMPhi3ForCausalLM


class PromptTemplates:
    """Handles prompt templates for different evaluation scenarios."""
    
    INSTRUCTION_PROMPT = """
Please answer questions based on the given text with format: "The {property} of {name} is {description}"
"""

    INSTRUCTION_PROMPT_MULTI = """
Please answer questions based on the given text with format: "The {property} of {name1} is {description}; The {property} of {name2} is {description}; ..."
"""

    ZERO_SHOT_PROMPT = """
Please answer the question in a very compact manner with format: The {property} of {name} is {description}
"""

    ZERO_SHOT_PROMPT_MULTI = """
Please answer the question in a very compact manner with format: "The {property} of {name1} is {description}; The {property} of {name2} is {description}; ...
"""


class TextProcessor:
    """Handles text processing for different model types."""
    
    @staticmethod
    def clean_llama_output(text: str) -> str:
        """Clean Llama model output by removing special tokens."""
        text = text.replace("<|eot_id|>", "")
        text = text.replace("<|start_header_id|>assistant<|end_header_id|>", "")
        text = text.replace("<|start_header_id|>user<|end_header_id|>", "")
        text = text.replace("<|end_of_text|>", "")
        return text

    @staticmethod
    def clean_phi3_output(text: str) -> str:
        """Clean Phi3 model output by removing special tokens."""
        text = text.replace("<|end|>", "")
        text = text.replace("<|assistant|>", "")
        text = text.replace("<|user|>", "")
        return text

    @staticmethod
    def format_llama_question(question: str) -> str:
        """Format question for Llama model."""
        return (
            "<|start_header_id|>user<|end_header_id|> "
            + question
            + "<|eot_id|>"
            + "<|start_header_id|>assistant<|end_header_id|>"
        )

    @staticmethod
    def format_phi3_question(question: str) -> str:
        """Format question for Phi3 model."""
        return "<|user|>\n" + question + "<|end|>\n" + "<|assistant|>\n"


class ModelHandler:
    """Handles model-specific operations and mappings."""
    
    # Model to question formatter mapping
    QUESTION_FORMATTERS = {
        KblamLlamaForCausalLM: TextProcessor.format_llama_question,
        AtlaskvLlamaForCausalLM: TextProcessor.format_llama_question,
        KBLaMPhi3ForCausalLM: TextProcessor.format_phi3_question,
    }
    
    # Model to output cleaner mapping
    OUTPUT_CLEANERS = {
        KblamLlamaForCausalLM: TextProcessor.clean_llama_output,
        AtlaskvLlamaForCausalLM: TextProcessor.clean_llama_output,
        KBLaMPhi3ForCausalLM: TextProcessor.clean_phi3_output,
    }

    @classmethod
    def get_question_formatter(cls, model_type):
        """Get question formatter for model type."""
        for model_class, formatter in cls.QUESTION_FORMATTERS.items():
            if isinstance(model_type, model_class):
                return formatter
        return None

    @classmethod
    def get_output_cleaner(cls, model_type):
        """Get output cleaner for model type."""
        for model_class, cleaner in cls.OUTPUT_CLEANERS.items():
            if isinstance(model_type, model_class):
                return cleaner
        return None


class MathUtils:
    """Handles mathematical operations."""
    
    @staticmethod
    def compute_softmax(scores: np.array, axis: int) -> np.array:
        """Compute softmax values for each set of scores."""
        e_x = np.exp(scores - np.max(scores))
        return e_x / e_x.sum(axis=axis)[:, None]


class QuestionAnswerer:
    """Handles question answering with different models."""
    
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, 
                 model: Union[KBLaMPhi3ForCausalLM, KblamLlamaForCausalLM, AtlaskvLlamaForCausalLM]):
        self.tokenizer = tokenizer
        self.model = model

    def answer_question(self, question: str, kb=None, kb_config: Optional[KBLaMConfig] = None) -> str:
        """Answer a question using the model."""
        # Format question for the specific model
        formatter = ModelHandler.get_question_formatter(self.model)
        if formatter:
            formatted_question = formatter(question)
        else:
            formatted_question = question

        # Tokenize input
        tokenizer_output = self.tokenizer(formatted_question, return_tensors="pt", padding=True).to("cuda")
        input_ids, attention_masks = tokenizer_output["input_ids"], tokenizer_output["attention_mask"]

        # Generate response
        with torch.autograd.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_masks,
                kb_kvs=kb,
                max_new_tokens=150,
                tokenizer=self.tokenizer,
                output_attentions=True,
                kb_config=kb_config,
            ).squeeze()

        # Decode and clean output
        decoded_output = self.tokenizer.decode(outputs, skip_special_tokens=False)
        cleaner = ModelHandler.get_output_cleaner(self.model)
        if cleaner:
            return cleaner(decoded_output)
        return decoded_output


# Backward compatibility functions
instruction_prompts = PromptTemplates.INSTRUCTION_PROMPT
instruction_prompts_multi_entities = PromptTemplates.INSTRUCTION_PROMPT_MULTI
zero_shot_prompt = PromptTemplates.ZERO_SHOT_PROMPT
zero_shot_prompt_multi_entities = PromptTemplates.ZERO_SHOT_PROMPT_MULTI

model_question_format_mapping = ModelHandler.QUESTION_FORMATTERS
model_prune_format_mapping = ModelHandler.OUTPUT_CLEANERS


def _prune_for_llama(S: str) -> str:
    """Prune Llama output (backward compatibility)."""
    return TextProcessor.clean_llama_output(S)


def _prune_for_phi3(S: str) -> str:
    """Prune Phi3 output (backward compatibility)."""
    return TextProcessor.clean_phi3_output(S)


def softmax(x: np.array, axis: int) -> np.array:
    """Compute softmax (backward compatibility)."""
    return MathUtils.compute_softmax(x, axis)


def _format_Q_llama(Q: str) -> str:
    """Format question for Llama (backward compatibility)."""
    return TextProcessor.format_llama_question(Q)


def _format_Q_phi3(Q: str) -> str:
    """Format question for Phi3 (backward compatibility)."""
    return TextProcessor.format_phi3_question(Q)


def answer_question(
    tokenizer: transformers.PreTrainedTokenizer,
    model: Union[KBLaMPhi3ForCausalLM, KblamLlamaForCausalLM, AtlaskvLlamaForCausalLM],
    Q: str,
    kb=None,
    kb_config: Optional[KBLaMConfig] = None,
) -> str:
    """Answer question using model (backward compatibility)."""
    answerer = QuestionAnswerer(tokenizer, model)
    return answerer.answer_question(Q, kb, kb_config)
