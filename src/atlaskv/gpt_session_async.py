import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from azure.identity import (
    AuthenticationRecord,
    DeviceCodeCredential,
    TokenCachePersistenceOptions,
    get_bearer_token_provider,
)
from openai import AsyncOpenAI


@dataclass
class AsyncSessionConfig:
    """Configuration for async GPT session."""
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


class AsyncModelValidator:
    """Handles async model validation and management."""
    
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


class AsyncClientManager:
    """Manages async OpenAI client initialization and configuration."""
    
    def __init__(self, config: AsyncSessionConfig):
        self.config = config
        self._setup_async_client()

    def _setup_async_client(self):
        """Setup async OpenAI client."""
        # Azure authentication setup
        token_provider = get_bearer_token_provider(
            self._get_credential(), "https://cognitiveservices.azure.com/.default"
        )

        # Azure client setup (commented out in original)
        # self.azure_client = AzureOpenAI(
        #     azure_endpoint=self.config.endpoint_url,
        #     api_version=self.config.api_version,
        #     azure_ad_token_provider=token_provider,
        # )

        self.openai_client = AsyncOpenAI(
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


class AsyncChatHandler:
    """Handles async chat completion operations."""
    
    def __init__(self, client: AsyncOpenAI, config: AsyncSessionConfig):
        self.client = client
        self.config = config

    async def make_chat_completion(self, messages: List[Dict]) -> Optional[str]:
        """Make async chat completion API call with retries."""
        for _ in range(self.config.max_retries):
            completion = await self.client.chat.completions.create(
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
                return completion.choices[0].message.content
        return None

    async def generate_response(self, prompt: str) -> Optional[str]:
        """Generate async response for a single prompt."""
        messages = [
            {"role": "system", "content": self.config.system_msg},
            {"role": "user", "content": prompt},
        ]
        return await self.make_chat_completion(messages)


class AsyncEmbeddingHandler:
    """Handles async embedding generation operations."""
    
    def __init__(self, client: AsyncOpenAI, config: AsyncSessionConfig):
        self.client = client
        self.config = config

    async def make_embedding_call(self, text: str) -> Optional[List[float]]:
        """Make async embedding API call with retries."""
        for _ in range(self.config.max_retries):
            embedding = await self.client.embeddings.create(
                input=text, model=self.config.model_name
            )
            if embedding:
                return embedding.data[0].embedding
        return None

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate async embedding for single text."""
        return await self.make_embedding_call(text)


class GPTAsync:
    """Alternative async GPT session with modular design."""
    
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
        AsyncModelValidator.validate_model(model_name)
        
        # Create configuration
        self.config = AsyncSessionConfig(
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
        self.client_manager = AsyncClientManager(self.config)
        self.chat_handler = AsyncChatHandler(self.client_manager.openai_client, self.config)
        self.embedding_handler = AsyncEmbeddingHandler(self.client_manager.openai_client, self.config)

    def set_seed(self, seed: int) -> None:
        """Set random seed for generation."""
        self.config.seed = seed

    async def api_call_chat(self, messages: List[Dict]) -> Optional[str]:
        """Make async chat completion API call."""
        return await self.chat_handler.make_chat_completion(messages)

    async def generate_response(self, prompt: str) -> Optional[str]:
        """Generate async response for given prompt."""
        return await self.chat_handler.generate_response(prompt)

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate async embedding for given text."""
        return await self.embedding_handler.generate_embedding(text)


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
    return parser


async def main():
    """Main async function for command line usage."""
    args = create_argument_parser().parse_args()
    gpt = GPTAsync(args.model_name, args.endpoint_url, args.endpoint_api_key)
    response = await gpt.generate_embedding(args.prompt)
    assert response is not None


if __name__ == "__main__":
    asyncio.run(main())