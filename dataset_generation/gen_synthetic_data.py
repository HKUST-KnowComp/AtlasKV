import argparse
import json
import os
import re
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from transformers import AutoModelForCausalLM

from atlaskv.gpt_session import GPT
from atlaskv.utils.data_utils import DataPoint, Entity, save_entity


@dataclass
class GenerationConfig:
    """Configuration for synthetic data generation."""
    model_name: str = "deepseek-chat"
    endpoint_url: str = ""
    endpoint_api_key: str = ""
    output_path: str = "your_output_path"
    generate_related_people: bool = True
    raw_output_file: str = "synthetic_data_raw.json"
    output_file: str = "synthetic_data_qkv.json"
    perturbed_output_file: str = "perturbed_output_file"
    augmented_output_file: str = "synthetic_data_qkv_augmented.json"
    token_log_file: str = "token_usage.jsonl"
    price_input_per_1k: Optional[float] = None
    price_output_per_1k: Optional[float] = None


class PromptBuilder:
    """Handles prompt construction and management."""
    
    def __init__(self):
        self.system_prompt = "You are a AI system that generates synthetic data examples in JSON format."
        self.entity_format_prompt = """
            \nMake sure to generate a single data point in the following JSON format:
            {
                "name": "{name}",
                "description": "{description}",
                "objectives": "{objectives}",
                "purpose": "{purpose}"
            }
        """
        self.prompt_2nd_phase = (
            """
            Now for each of the names generated, generate a short desciption, short objectives, and a purpose for the data point.
            Please ensure that the generated contents has **LOW** correlation with the name.
            Make the data point styles diverse using a mixture of formal and informal language.
        """
            + self.entity_format_prompt
            + " Do **NOT** generate anything else."
        )

    def build_qa_prompts(self, entity: DataPoint) -> Tuple[str, str, str]:
        """Build question, answer, and key string from entity."""
        question = f"What is the {entity.description_type} of {entity.name}?"
        answer = f"The {entity.description_type} of {entity.name} is {entity.description}."
        key_string = f"the {entity.description_type} of {entity.name}"
        return question, answer, key_string

    def build_entity_generation_prompt(self, instruction: str) -> List[Dict[str, str]]:
        """Build prompt for entity generation."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": instruction},
        ]

    def build_entity_completion_prompt(self, instruction: str, initial_output: str) -> List[Dict[str, str]]:
        """Build prompt for entity completion."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": initial_output},
            {"role": "user", "content": self.prompt_2nd_phase},
        ]

    def build_related_data_prompt(self, entity: Entity) -> str:
        """Build prompt for related data generation."""
        instruction = f"Generate a person name related to the entity {entity.name} with description {entity.description}."
        instruction += "The person needs to be associated with the entity in some way. e.g. they work in the company or they are a character in the book."
        instruction += f"Make sure the entity is in the format of {self.entity_format_prompt}"
        return instruction

    def build_augmentation_prompt(self, data: DataPoint) -> str:
        """Build prompt for data augmentation."""
        return (
            "Generate an extended Q and an A for this pair: "
            + f"Q: {data.Q}\nA: {data.A}" + "**Please generate in the format of**:\nQ: ... \nA: ..."
        )

    def build_perturbation_prompt(self, data: DataPoint) -> str:
        """Build prompt for name perturbation."""
        prompt = f"Perturb the names in the queries of the dataset (e.g. Margaret Thatcher -> Maggie Thatcher or Microsoft Research to MSR) for data point with name {data.name}."
        prompt += f"Return the question {data.Q} with the perturbed name. Make sure the perturbation is valid. Do NOT generate anything else."
        return prompt


class DataSourceManager:
    """Manages data sources and types for generation."""
    
    def __init__(self):
        self.idea_sources = [
            "software companies", "tech companies", "software tools", "greek letters",
            "product reviews", "product releases", "work-related concepts", "work-related documents",
            "document types", "financial terms", "legal terms", "medical terms",
            "fiction characters", "famous rock bands", "birds", "animals",
            "natural phenomena", "physical locations", "artist names", "classical music",
            "musical instruments", "music genres", "art styles", "ancient Roman concepts",
            "Hindu myths", "Cthulhu Mythos", "real-world company names", "mythological creatures",
            "planets and stars", "historical figures", "political figures", "literary genres",
            "botanical names", "famous landmarks", "scientific concepts", "space missions",
            "inventions", "philosophical terms", "chemical elements", "famous scientists",
            "famous mathematicians", "famous authors", "marine life", "mythological places",
            "famous battles", "sports teams", "sport events", "food and drinks",
        ]

        self.data_types = [
            "person name", "idea", "team", "meeting", "event", "location", "document",
            "presentation", "meeting", "conference", "workshop", "database", "organization",
            "tech company", "car company", "entertainment company", "construction company",
            "retail company", "finance company", "healthcare company", "restaurant",
            "hotel", "museum", "university", "educational institution", "government agency",
            "hospital", "github repo", "project", "meeting room", "building", "product",
            "lab", "airline", "textbook", "tv show", "music album", "website", "personal blog",
            "gaming company", "game", "movie studio", "consulting firm", "biotech company",
            "app", "software tool", "bookstore", "coffee shop", "bar", "e-commerce site",
            "social media platform", "fitness brand", "fashion brand", "beauty brand",
            "food brand", "drink brand", "sports brand", "travel brand",
            "non-profit organization", "political party",
        ]

    def get_generation_instructions(self) -> List[str]:
        """Get list of generation instructions."""
        return [
            f"Please randomly generate a {name_type} name innovated by or associated with {idea_type}."
            "The generated name should be of diverse style and length. A valid name should consist of a single word (such as Alexandria or Microsoft) or multiple words (such as Microsoft Office or Theta-Phoenix Entertainment). "
            for (name_type, idea_type) in product(self.idea_sources, self.data_types)
        ]


class EntityProcessor:
    """Handles entity processing and transformation."""
    
    def __init__(self, prompt_builder: PromptBuilder):
        self.prompt_builder = prompt_builder

    def parse_entity_from_response(self, response: str) -> Entity:
        """Parse entity from GPT response."""
        cleaned_response = response.replace("```", "").replace("json", "").strip()
        return Entity(**json.loads(cleaned_response))

    def process_entities_to_datapoints(self, entity_list: List[Entity]) -> List[DataPoint]:
        """Convert entities to datapoints."""
        dataset = []
        keywords = {"description", "objectives", "purpose"}

        for entity in entity_list:
            for keyword in keywords:
                datapoint = DataPoint(
                    name=entity.name,
                    description_type=keyword.lower(),
                    description=getattr(entity, keyword),
                )
                datapoint.Q, datapoint.A, datapoint.key_string = self.prompt_builder.build_qa_prompts(datapoint)
                dataset.append(datapoint)

        return dataset

    def augment_with_extended_qa(self, dataset: List[DataPoint], generator: 'SyntheticDataGenerator') -> List[DataPoint]:
        """Augment dataset with extended Q&A."""
        generator.system_prompt = """You are given a question and answer pair, please extend the question to be open-ended and generate a short answer. 
                                For example, you could generate "What is the objective of xxx and what do you think of it?"
                                Make sure the answer is **only** based on information provided from the QA pair. In addition, please generate in the format of:
                                Q: ...
                                A: ... 
                            """

        for data in tqdm(dataset):
            try:
                prompt = self.prompt_builder.build_augmentation_prompt(data)
                gpt_output = generator.generate_response(prompt)
                extended_q = re.findall(r"Q: (.*)", gpt_output)[0]
                extended_a = re.findall(r"A: (.*)", gpt_output)[0]
                data.extended_Q = extended_q
                data.extended_A = extended_a
            except Exception as e:
                print("Error augmenting Q&A.")
                print(e)
                continue
        return dataset

    def perturb_names(self, dataset: List[DataPoint], generator: 'SyntheticDataGenerator') -> List[DataPoint]:
        """Perturb names in the dataset."""
        for data in tqdm(dataset):
            try:
                prompt = self.prompt_builder.build_perturbation_prompt(data)
                gpt_output = generator.generate_response(prompt)
                data.Q = gpt_output
            except Exception as e:
                print("Error perturbing the names in the queries.")
                print(e)
                continue
        return dataset


class SyntheticDataGenerator(GPT):
    """Main synthetic data generator with refactored structure."""
    
    def __init__(
        self, 
        model: str, 
        endpoint_url: str, 
        endpoint_api_key: str, 
        **kwargs
    ) -> None:
        super().__init__(model, endpoint_url, endpoint_api_key=endpoint_api_key, **kwargs)
        
        self.prompt_builder = PromptBuilder()
        self.source_manager = DataSourceManager()
        self.entity_processor = EntityProcessor(self.prompt_builder)

    def generate_entity(self, instruction: str) -> Entity:
        """Generate a single entity from instruction."""
        prompt = self.prompt_builder.build_entity_generation_prompt(instruction)
        gpt_output = self.api_call_chat(prompt)

        messages = self.prompt_builder.build_entity_completion_prompt(instruction, gpt_output)
        print(messages)
        gpt_output = self.api_call_chat(messages)
        return self.entity_processor.parse_entity_from_response(gpt_output)

    def generate_related_data(self, entity: Entity) -> Entity:
        """Generate related data for an entity."""
        instruction = self.prompt_builder.build_related_data_prompt(entity)
        prompt = [
            {"role": "system", "content": self.prompt_builder.system_prompt},
            {"role": "user", "content": instruction},
        ]
        print(prompt)
        gpt_output = self.api_call_chat(prompt)
        return self.entity_processor.parse_entity_from_response(gpt_output)

    def get_instructions(self) -> List[str]:
        """Get generation instructions."""
        return self.source_manager.get_generation_instructions()

    def post_process_data(self, entity_list: List[Entity]) -> List[DataPoint]:
        """Post-process entities into datapoints."""
        return self.entity_processor.process_entities_to_datapoints(entity_list)

    def augmenta_data_with_synthetic_QA(self, dataset: List[DataPoint]) -> List[DataPoint]:
        """Augment dataset with synthetic Q&A."""
        return self.entity_processor.augment_with_extended_qa(dataset, self)

    def perturb_names(self, dataset: List[DataPoint]) -> List[DataPoint]:
        """Perturb names in dataset."""
        return self.entity_processor.perturb_names(dataset, self)


class DataPipeline:
    """Orchestrates the entire data generation pipeline."""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.generator = SyntheticDataGenerator(
            config.model_name, 
            config.endpoint_url, 
            config.endpoint_api_key
        )

    def setup_token_logging(self):
        """Setup token logging if configured."""
        if self.config.price_input_per_1k is not None and self.config.price_output_per_1k is not None:
            price_map = {
                self.config.model_name: {
                    "input": self.config.price_input_per_1k, 
                    "output": self.config.price_output_per_1k
                }
            }
            token_log_path = os.path.join(self.config.output_path, self.config.token_log_file)
            self.generator.set_token_logging(log_path=token_log_path, price_per_1k=price_map)

    def load_or_generate_entities(self) -> List[Entity]:
        """Load existing entities or generate new ones."""
        raw_output_file = os.path.join(self.config.output_path, self.config.raw_output_file)
        
        if os.path.exists(raw_output_file):
            with open(raw_output_file, "r") as file:
                return [Entity(**json.loads(line)) for line in file]
        else:
            return self._generate_new_entities(raw_output_file)

    def _generate_new_entities(self, raw_output_file: str) -> List[Entity]:
        """Generate new entities."""
        entity_list = []
        for seed in range(1):
            self.generator.set_seed(seed)
            for instruction in tqdm(self.generator.get_instructions()):
                for i in range(50):
                    try:
                        entity = self.generator.generate_entity(instruction)
                    except Exception as e:
                        print("Error generating entity.")
                        print(e)
                        continue
                    save_entity(entity, raw_output_file)
                    entity_list.append(entity)

                    if self.config.generate_related_people:
                        try:
                            response = self.generator.generate_related_data(entity)
                        except Exception as e:
                            print("Error generating entity.")
                            print(e)
                            continue
                        save_entity(response, raw_output_file)
                        entity_list.append(response)
        return entity_list

    def load_or_generate_dataset(self, entity_list: List[Entity]) -> List[DataPoint]:
        """Load existing dataset or generate new one."""
        QA_output_file = os.path.join(self.config.output_path, self.config.output_file)
        
        if os.path.exists(QA_output_file):
            with open(QA_output_file, "r") as file:
                return [DataPoint(**json.loads(line)) for line in file]
        else:
            dataset = self.generator.post_process_data(entity_list)
            for data in dataset:
                save_entity(data, QA_output_file)
            return dataset

    def process_perturbed_dataset(self, dataset: List[DataPoint]) -> List[DataPoint]:
        """Process dataset with name perturbation."""
        perturbed_dataset = self.generator.perturb_names(dataset)
        for data in perturbed_dataset:
            save_entity(data, os.path.join(self.config.output_path, self.config.perturbed_output_file))
        return perturbed_dataset

    def process_augmented_dataset(self, dataset: List[DataPoint]) -> List[DataPoint]:
        """Process dataset with augmentation."""
        augmented_dataset = self.generator.augmenta_data_with_synthetic_QA(dataset)
        for data in augmented_dataset:
            save_entity(data, os.path.join(self.config.output_path, self.config.augmented_output_file))
        return augmented_dataset

    def write_token_summary(self):
        """Write token usage summary."""
        try:
            summary = self.generator.get_token_usage_summary()
            with open(os.path.join(self.config.output_path, "token_usage_summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("Error writing token usage summary:", e)

    def run_pipeline(self):
        """Run the complete data generation pipeline."""
        os.makedirs(self.config.output_path, exist_ok=True)
        self.setup_token_logging()

        # Generate or load entities
        entity_list = self.load_or_generate_entities()

        # Generate or load dataset
        dataset = self.load_or_generate_dataset(entity_list)

        # Process perturbed dataset
        self.process_perturbed_dataset(dataset)

        # Process augmented dataset
        self.process_augmented_dataset(dataset)

        # Write token usage summary
        self.write_token_summary()


def create_config_from_args(args: argparse.Namespace) -> GenerationConfig:
    """Create configuration from command line arguments."""
    return GenerationConfig(
        model_name=args.model_name,
        endpoint_url=args.endpoint_url,
        endpoint_api_key=args.endpoint_api_key,
        output_path=args.output_path,
        generate_related_people=args.generate_related_people,
        raw_output_file=args.raw_output_file,
        output_file=args.output_file,
        perturbed_output_file=args.perturbed_output_file,
        augmented_output_file=args.augmented_output_file,
        token_log_file=args.token_log_file,
        price_input_per_1k=args.price_input_per_1k,
        price_output_per_1k=args.price_output_per_1k,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="deepseek-chat")
    parser.add_argument("--endpoint_url", type=str, required=True)
    parser.add_argument("--endpoint_api_key", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="your_output_path")
    parser.add_argument("--generate_related_people", type=bool, default=True)
    parser.add_argument("--raw_output_file", type=str, default="synthetic_data_raw.json")
    parser.add_argument("--output_file", type=str, default="synthetic_data_qkv.json")
    parser.add_argument("--perturbed_output_file", type=str, default="perturbed_output_file")
    parser.add_argument("--augmented_output_file", type=str, default="synthetic_data_qkv_augmented.json")
    parser.add_argument("--token_log_file", type=str, default="token_usage.jsonl")
    parser.add_argument("--price_input_per_1k", type=float, default=None)
    parser.add_argument("--price_output_per_1k", type=float, default=None)
    return parser


def main():
    """Main function."""
    args = build_parser().parse_args()
    config = create_config_from_args(args)
    
    pipeline = DataPipeline(config)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
