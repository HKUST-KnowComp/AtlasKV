import asyncio
import csv
import json
import random
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm


@dataclass
class AtlasConfig:
    """Configuration for Atlas training data generation."""
    api_key: str = ""
    base_url: str = ""
    model_name: str = ""
    lkg_dir: str = ""
    lkg_file: str = ""
    output_folder: str = ""
    output_file: str = ""
    max_tokens: int = 0
    min_tokens: int = 0
    max_name_tokens: int = 0
    max_name_chars: int = 0
    min_description_tokens: int = 0
    max_description_tokens: int = 0
    max_description_chars: int = 0


class ConfigLoader:
    """Handles configuration loading and management."""
    
    @staticmethod
    def load_config(config_path: str) -> AtlasConfig:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        return AtlasConfig(
            api_key=config['openai']['api_key'],
            base_url=config['openai']['base_url'],
            model_name=config['openai']['chat_model'],
            lkg_dir=config['kg2kv']['lkg_dir'],
            lkg_file=config['kg2kv']['lkg_file'],
            output_folder=config['kg2kv']['output_folder'],
            output_file=config['kg2kv']['output_file'],
            max_tokens=config['kg2kv']['max_tokens'],
            min_tokens=config['kg2kv']['min_tokens'],
            max_name_tokens=config['kg2kv']['max_name_tokens'],
            max_name_chars=config['kg2kv']['max_name_tokens'] * 20,
            min_description_tokens=config['kg2kv']['min_description_tokens'],
            max_description_tokens=config['kg2kv']['max_description_tokens'],
            max_description_chars=config['kg2kv']['max_description_tokens'] * 20,
        )


class TextAnalyzer:
    """Handles text analysis operations."""
    
    @staticmethod
    def count_tokens(text: str) -> int:
        """Count tokens in text (simple word count)."""
        return len(text.split())

    @staticmethod
    def count_chars(text: str) -> int:
        """Count characters in text."""
        return len(text)


class RelationProcessor:
    """Handles relation processing and natural language conversion."""
    
    def __init__(self, config: AtlasConfig):
        self.config = config
        self.relation_prompt = self._build_relation_prompt()
        self.relation_prompt_short = self._build_short_relation_prompt()

    def _build_relation_prompt(self) -> str:
        """Build the full relation conversion prompt."""
        return """
**Task:** Convert a relation phrase to its natural noun equivalent based on entity position.

**Input:**
- `relation`: A verb phrase (e.g., "is participated by", "produces")
- `position`: Either "head entity/event" (missing head) or "tail entity/event" (missing tail)

**Conversion Rules:**
1. **For missing head entity** (position = "head entity/event"):
   - If relation matches predefined pattern → Use corresponding agent noun:
     ```
     "is participated by" → "participation"
     "has purpose"        → "origin"
     ```
   - For passive relations ("is [verb]ed by"):
     - Remove "is " and " by"
     - Convert verb to agent noun:
       - Add "-er"/"-or" (e.g., "govern" → "governor")
       - Add "-ist" (e.g., "specialize" → "specialist")
       - Use irregular forms (e.g., "advise" → "advisor")

2. **For missing tail entity** (position = "tail entity/event"):
   - If relation matches predefined pattern → Use corresponding object noun:
     ```
     "is participated by" → "participant"
     "has purpose"        → "purpose"
     ```
   - For active relations (e.g., "[verb]s"):
     - Remove trailing "s" (if present)
     - Convert verb to object noun:
       - Add "-tion"/"-sion" (e.g., "produce" → "production")
       - Add "-ment" (e.g., "achieve" → "achievement")
       - Use irregular forms (e.g., "sell" → "sale")

**Output:** Only the natural noun (no explanations)

**Examples:**
- Input: `("is participated by", "head entity/event")`, Output: "participation"
- Input: `("is participated by", "tail entity/event")`, Output: "participant"
- Input: `("produces", "head entity/event")`, Output: "producer"
- Input: `("produces", "tail entity/event")`, Output: "product"
"""

    def _build_short_relation_prompt(self) -> str:
        """Build the short relation conversion prompt."""
        return """
**Task:** Convert relation phrase to natural noun based on missing entity position.

**Rules:**
- **Missing head**: Passive relations → agent nouns ("is governed by" → "governor", "is participated by" → "participation")
- **Missing tail**: Active relations → object nouns ("produces" → "product", "achieves" → "achievement")

**Output:** Natural noun only.

**Examples:**
- ("is participated by", "head") → "participation"
- ("is participated by", "tail") → "participant"  
- ("produces", "head") → "producer"
- ("produces", "tail") → "product"
"""

    async def convert_relation_to_nl(self, client: AsyncOpenAI, semaphore: asyncio.Semaphore, 
                                   relation: str, missing: str = 'head') -> str:
        """Convert relation to natural language using OpenAI API."""
        async with semaphore:
            response = await client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": self.relation_prompt},
                    {"role": "user", "content": f"relation: {relation}, missing: {missing}"}
                ]
            )
            return response.choices[0].message.content.strip()


class TripleProcessor:
    """Handles triple processing and filtering."""
    
    def __init__(self, config: AtlasConfig):
        self.config = config
        self.analyzer = TextAnalyzer()

    def should_skip_triple(self, head: str, tail: str, relation: str) -> bool:
        """Check if triple should be skipped based on content."""
        return "participate" in relation

    def is_entity_duplicate(self, entity: str, seen_entities: Set[str]) -> bool:
        """Check if entity has already been processed."""
        return entity in seen_entities

    def validate_triple_lengths(self, head: str, tail: str, missing: str) -> bool:
        """Validate triple lengths based on missing entity position."""
        if missing == 'head':
            if (self.analyzer.count_tokens(tail) >= self.config.max_name_tokens or 
                self.analyzer.count_chars(tail) >= self.config.max_name_chars):
                return False
            if (self.analyzer.count_tokens(head) <= self.config.min_description_tokens or 
                self.analyzer.count_tokens(head) >= self.config.max_description_tokens or 
                self.analyzer.count_chars(head) >= self.config.max_description_chars):
                return False
        else:
            if (self.analyzer.count_tokens(head) >= self.config.max_name_tokens or 
                self.analyzer.count_chars(head) >= self.config.max_name_chars):
                return False
            if (self.analyzer.count_tokens(tail) <= self.config.min_description_tokens or 
                self.analyzer.count_tokens(tail) >= self.config.max_description_tokens or 
                self.analyzer.count_chars(tail) >= self.config.max_description_chars):
                return False
        return True

    def process_triples_from_file(self, max_triples: int) -> List[Dict[str, str]]:
        """Process triples from CSV file."""
        triples_to_process = []
        seen_entities = set()
        
        with open(f"{self.config.lkg_dir}/{self.config.lkg_file}", 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            for row in reader:
                if len(triples_to_process) >= max_triples:
                    break
                
                head, tail, relation = row[0], row[1], row[2]

                if self.should_skip_triple(head, tail, relation):
                    continue

                if (self.is_entity_duplicate(head, seen_entities) or 
                    self.is_entity_duplicate(tail, seen_entities)):
                    continue

                missing = random.choice(['head', 'tail'])
                
                if not self.validate_triple_lengths(head, tail, missing):
                    continue
                
                seen_entities.add(head)
                seen_entities.add(tail)
                
                triple_info = {
                    'head': head, 
                    'tail': tail, 
                    'relation': relation, 
                    'missing': missing
                }
                triples_to_process.append(triple_info)
                print(f"Processed triples: {len(triples_to_process)}", end='\r', flush=True)

        return triples_to_process


class DataGenerator:
    """Handles data generation and formatting."""
    
    def __init__(self, config: AtlasConfig):
        self.config = config
        self.analyzer = TextAnalyzer()

    def create_data_item(self, head: str, tail: str, missing: str, relation_nl: str) -> Optional[Dict[str, str]]:
        """Create a data item from triple information."""
        name = f"{tail}" if missing == 'head' else f"{head}"
        key_string = f"the {relation_nl} of {name}"
        
        if self.analyzer.count_tokens(key_string) > self.config.max_tokens + 2:
            return None
        
        question = f"What is {key_string}?"
        answer = f"{key_string} is {head}" if missing == 'head' else f"{key_string} is {tail}"
        description_type = f"{relation_nl}"
        description = f"{head}" if missing == 'head' else f"{tail}"
        
        return {
            "name": name,
            "description_type": description_type,
            "description": description,
            "Q": question,
            "A": answer,
            "key_string": key_string,
            "extended_Q": "",
            "extended_A": ""
        }

    def save_triples_intermediate(self, triples: List[Dict[str, str]]) -> None:
        """Save triples to intermediate file."""
        with open(f"{self.config.output_folder}/triples_to_process.json", 'w', encoding='utf-8') as f:
            json.dump(triples, f, ensure_ascii=False, indent=2)

    def load_triples_intermediate(self) -> List[Dict[str, str]]:
        """Load triples from intermediate file."""
        with open(f"{self.config.output_folder}/triples_to_process.json", 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_final_results(self, results: List[Dict[str, str]]) -> None:
        """Save final results to output file."""
        random.shuffle(results)
        with open(self.config.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


class AtlasDataBuilder:
    """Main orchestrator for Atlas training data generation."""
    
    def __init__(self, config: AtlasConfig):
        self.config = config
        self.triple_processor = TripleProcessor(config)
        self.relation_processor = RelationProcessor(config)
        self.data_generator = DataGenerator(config)

    async def process_relations_async(self, triples: List[Dict[str, str]]) -> List[str]:
        """Process relations asynchronously to get natural language equivalents."""
        semaphore = asyncio.Semaphore(256)
        async with AsyncOpenAI(api_key=self.config.api_key, base_url=self.config.base_url) as client:
            tasks = [
                self.relation_processor.convert_relation_to_nl(
                    client, semaphore, triple['relation'], triple['missing']
                ) 
                for triple in triples
            ]
            return await tqdm.gather(*tasks, desc="Generating NL relations")

    def generate_final_dataset(self, triples: List[Dict[str, str]], relation_nls: List[str]) -> List[Dict[str, str]]:
        """Generate final dataset from triples and relation NLs."""
        results = []
        
        for i, triple_info in enumerate(triples):
            head = triple_info['head']
            tail = triple_info['tail']
            missing = triple_info['missing']
            relation_nl = relation_nls[i]

            data_item = self.data_generator.create_data_item(head, tail, missing, relation_nl)
            if data_item:
                results.append(data_item)
        
        return results

    async def build_dataset(self, max_triples: int = 150000) -> None:
        """Build the complete Atlas training dataset."""
        # Process triples from file
        triples = self.triple_processor.process_triples_from_file(max_triples)
        
        # Save intermediate results
        self.data_generator.save_triples_intermediate(triples)
        
        # Load triples (for consistency with original)
        triples = self.data_generator.load_triples_intermediate()
        
        # Process relations asynchronously
        relation_nls = await self.process_relations_async(triples)
        
        # Generate final dataset
        results = self.generate_final_dataset(triples, relation_nls)
        
        # Save final results
        self.data_generator.save_final_results(results)


def load_config_from_file(config_path: str) -> AtlasConfig:
    """Load configuration from YAML file."""
    return ConfigLoader.load_config(config_path)


async def main():
    """Main function."""
    config_path = '../config/training_data_atlas_config.yaml'
    config = load_config_from_file(config_path)
    
    builder = AtlasDataBuilder(config)
    await builder.build_dataset(max_triples=150000)


if __name__ == "__main__":
    asyncio.run(main())
