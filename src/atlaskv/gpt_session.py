import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from azure.identity import (
    AuthenticationRecord,
    DeviceCodeCredential,
    TokenCachePersistenceOptions,
    get_bearer_token_provider,
)
from openai import AzureOpenAI, OpenAI


@dataclass
class SessionConfig:
    """Configuration for GPT session."""
    model_name: str = ""
    endpoint_url: str = ""
    endpoint_api_key: str = ""
    api_version: str = "2024-02-15-preview"
    system_msg: str = "You are an AI assistant."
    max_retries: int = 12
    temperature: float = 1.0
    max_tokens: int = 4096
    top_p: float = 0.95
    frequency_penalty: int = 0
    presence_penalty: int = 0
    seed: Optional[int] = None


class ModelValidator:
    """Handles model validation and management."""
    
    SUPPORTED_MODELS = [
        "gpt-4o", "ada-embeddings", "text-embedding-3-large", 
        "meta-llama/Meta-Llama-3.1-8B-Instruct", "deepseek-ai/DeepSeek-V3", 
        "deepseek-chat", "text-embedding-ada-002"
    ]

    @classmethod
    def validate_model(cls, model_name: str) -> None:
        """Validate model name against supported models."""
        if model_name not in cls.SUPPORTED_MODELS:
            raise ValueError(
                f"Invalid model: {model_name}. Valid models are: {cls.SUPPORTED_MODELS}"
            )


class ClientManager:
    """Manages OpenAI client initialization and configuration."""
    
    def __init__(self, config: SessionConfig):
        self.config = config
        self._setup_clients()

    def _setup_clients(self):
        """Setup OpenAI and Azure clients."""
        # Azure authentication setup (commented out in original)
        # token_provider = get_bearer_token_provider(
        #     self._get_credential(), "https://cognitiveservices.azure.com/.default"
        # )
        
        # self.azure_client = AzureOpenAI(
        #     azure_endpoint=self.config.endpoint_url,
        #     api_version=self.config.api_version,
        #     azure_ad_token_provider=token_provider,
        # )

        self.openai_client = OpenAI(
            api_key=self.config.endpoint_api_key,
            base_url=self.config.endpoint_url,
        )

    def _get_credential(self, lib_name: str = "azure_openai") -> DeviceCodeCredential:
        """Retrieve Azure authentication credential."""
        if sys.platform.startswith("win"):
            auth_record_root_path = Path(os.environ["LOCALAPPDATA"])
        else:
            auth_record_root_path = Path.home()

        auth_record_path = auth_record_root_path / lib_name / "auth_record.json"
        cache_options = TokenCachePersistenceOptions(
            name=f"{lib_name}.cache", allow_unencrypted_storage=True
        )

        if auth_record_path.exists():
            with open(auth_record_path, "r") as f:
                record_json = f.read()
            deserialized_record = AuthenticationRecord.deserialize(record_json)
            credential = DeviceCodeCredential(
                authentication_record=deserialized_record,
                cache_persistence_options=cache_options,
            )
        else:
            auth_record_path.parent.mkdir(parents=True, exist_ok=True)
            credential = DeviceCodeCredential(cache_persistence_options=cache_options)
            record_json = credential.authenticate().serialize()
            with open(auth_record_path, "w") as f:
                f.write(record_json)

        return credential


class TokenUsageTracker:
    """Handles token usage tracking and logging."""
    
    def __init__(self):
        self._token_logs: List[Dict] = []
        self._token_totals: Dict = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }
        self._price_per_1k: Optional[Dict] = None
        self._token_log_path: Optional[str] = None

    def configure_logging(self, log_path: Optional[str] = None, price_per_1k: Optional[Dict] = None) -> None:
        """Configure token usage logging."""
        self._token_log_path = log_path
        self._price_per_1k = price_per_1k

    def record_usage(self, kind: str, model: str, usage_obj) -> None:
        """Record token usage for a single call."""
        if usage_obj is None:
            return

        prompt_tokens = getattr(usage_obj, "prompt_tokens", None)
        completion_tokens = getattr(usage_obj, "completion_tokens", None)
        total_tokens = getattr(usage_obj, "total_tokens", None)

        cost_usd = self._calculate_cost(model, prompt_tokens, completion_tokens)

        log_entry = {
            "type": kind,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
        }

        self._update_totals(prompt_tokens, completion_tokens, total_tokens, cost_usd)
        self._token_logs.append(log_entry)
        self._write_log_entry(log_entry)

    def _calculate_cost(self, model: str, prompt_tokens: Optional[int], completion_tokens: Optional[int]) -> Optional[float]:
        """Calculate cost based on pricing configuration."""
        if not self._price_per_1k or model not in self._price_per_1k:
            return None
        
        if prompt_tokens is None or completion_tokens is None:
            return None
        
        pricing = self._price_per_1k[model]
        input_price = pricing.get("input")
        output_price = pricing.get("output")
        
        if input_price is None or output_price is None:
            return None
        
        return (prompt_tokens / 1000.0) * input_price + (completion_tokens / 1000.0) * output_price

    def _update_totals(self, prompt_tokens: Optional[int], completion_tokens: Optional[int], 
                      total_tokens: Optional[int], cost_usd: Optional[float]) -> None:
        """Update cumulative token totals."""
        if prompt_tokens is not None:
            self._token_totals["prompt_tokens"] += prompt_tokens
        if completion_tokens is not None:
            self._token_totals["completion_tokens"] += completion_tokens
        if total_tokens is not None:
            self._token_totals["total_tokens"] += total_tokens
        if cost_usd is not None:
            self._token_totals["cost_usd"] += cost_usd

    def _write_log_entry(self, log_entry: Dict) -> None:
        """Write log entry to file if configured."""
        if not self._token_log_path:
            return
        
        try:
            os.makedirs(os.path.dirname(self._token_log_path), exist_ok=True)
            with open(self._token_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def get_usage_summary(self) -> Dict:
        """Get cumulative token usage summary."""
        return dict(self._token_totals)


class ChatHandler:
    """Handles chat completion operations."""
    
    def __init__(self, client: OpenAI, config: SessionConfig, token_tracker: TokenUsageTracker):
        self.client = client
        self.config = config
        self.token_tracker = token_tracker

    def make_chat_completion(self, messages: List[Dict]) -> Optional[str]:
        """Make chat completion API call with retries."""
        for _ in range(self.config.max_retries):
            completion = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                seed=self.config.seed if self.config.seed else None,
            )
            if completion:
                usage_obj = getattr(completion, "usage", None)
                self.token_tracker.record_usage("chat", self.config.model_name, usage_obj)
                return completion.choices[0].message.content
        return None

    def generate_response(self, prompt: str) -> Optional[str]:
        """Generate response for a single prompt."""
        messages = [
            {"role": "system", "content": self.config.system_msg},
            {"role": "user", "content": prompt},
        ]
        return self.make_chat_completion(messages)


class EmbeddingHandler:
    """Handles embedding generation operations."""
    
    def __init__(self, client: OpenAI, config: SessionConfig, token_tracker: TokenUsageTracker):
        self.client = client
        self.config = config
        self.token_tracker = token_tracker

    def make_embedding_call(self, text: str) -> Optional[List[float]]:
        """Make embedding API call with retries."""
        for _ in range(self.config.max_retries):
            embedding = self.client.embeddings.create(
                input=text, model=self.config.model_name
            )
            if embedding:
                usage_obj = getattr(embedding, "usage", None)
                self.token_tracker.record_usage("embeddings", self.config.model_name, usage_obj)
                return embedding.data[0].embedding
        return None

    def make_batch_embedding_call(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Make batch embedding API call with retries."""
        for _ in range(self.config.max_retries):
            embedding = self.client.embeddings.create(
                input=texts, model=self.config.model_name
            )
            if embedding:
                return [data.embedding for data in embedding.data]
        return None

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for single text."""
        return self.make_embedding_call(text)

    def generate_embeddings_batch(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Generate embeddings for batch of texts."""
        if not texts:
            return []
        return self.make_batch_embedding_call(texts)


class GPT:
    """Alternative GPT session with modular design."""
    
    def __init__(
        self,
        model_name: str,
        endpoint_url: str,
        endpoint_api_key: str,
        api_version: str = "2024-02-15-preview",
        system_msg: str = "You are an AI assistant.",
        max_retries: int = 12,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        frequency_penalty: int = 0,
        presence_penalty: int = 0,
        seed: Optional[int] = None,
    ):
        # Validate model
        ModelValidator.validate_model(model_name)
        
        # Create configuration
        self.config = SessionConfig(
            model_name=model_name,
            endpoint_url=endpoint_url,
            endpoint_api_key=endpoint_api_key,
            api_version=api_version,
            system_msg=system_msg,
            max_retries=max_retries,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
        )
        
        # Initialize components
        self.client_manager = ClientManager(self.config)
        self.token_tracker = TokenUsageTracker()
        self.chat_handler = ChatHandler(self.client_manager.openai_client, self.config, self.token_tracker)
        self.embedding_handler = EmbeddingHandler(self.client_manager.openai_client, self.config, self.token_tracker)

    def set_token_logging(self, log_path: Optional[str] = None, price_per_1k: Optional[Dict] = None) -> None:
        """Enable per-call token usage logging."""
        self.token_tracker.configure_logging(log_path, price_per_1k)

    def get_token_usage_summary(self) -> Dict:
        """Return cumulative token usage and estimated cost."""
        return self.token_tracker.get_usage_summary()

    def set_seed(self, seed: int) -> None:
        """Set random seed for generation."""
        self.config.seed = seed

    def api_call_chat(self, messages: List[Dict]) -> Optional[str]:
        """Make chat completion API call."""
        return self.chat_handler.make_chat_completion(messages)

    def generate_response(self, prompt: str) -> Optional[str]:
        """Generate response for given prompt."""
        return self.chat_handler.generate_response(prompt)

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for given text."""
        return self.embedding_handler.generate_embedding(text)

    def generate_embeddings_batch(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Generate embeddings for batch of texts."""
        return self.embedding_handler.generate_embeddings_batch(texts)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(description="GPT Session")
    parser.add_argument(
        "--model_name",
        type=str,
        default="ada-embeddings",
        help="Model name to use for embedding generation",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Embedding text",
        help="Prompt for text generation",
    )
    parser.add_argument(
        "--endpoint_url",
        type=str,
        help="Endpoint URL for the model",
    )
    parser.add_argument(
        "--endpoint_api_key",
        type=str,
        help="API key for the endpoint",
    )
    parser.add_argument(
        "--batch_mode",
        action="store_true",
        help="Enable batch mode for processing multiple texts",
    )
    return parser


def main():
    """Main function for command line usage."""
    args = create_argument_parser().parse_args()
    gpt = GPT(args.model_name, args.endpoint_url, args.endpoint_api_key)
    response = gpt.generate_embedding(args.prompt)
    assert response is not None


if __name__ == "__main__":
    main()