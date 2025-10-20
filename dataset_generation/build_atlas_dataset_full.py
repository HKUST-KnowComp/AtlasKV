import asyncio
import csv
import glob
import json
import os
import random
import subprocess
import psutil
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from openai import AsyncOpenAI
from tqdm import tqdm as normal_tqdm
from tqdm.asyncio import tqdm


@dataclass
class FullAtlasConfig:
    """Configuration for full Atlas dataset generation."""
    api_key: str = ""
    base_url: str = ""
    model_name: str = ""
    lkg_dir: str = ""
    lkg_file: str = ""
    output_folder: str = ""
    output_file: str = ""
    if_check: bool = True
    if_naive: bool = False
    max_name_tokens: int = 0
    max_name_chars: int = 0
    max_description_tokens: int = 0
    max_description_chars: int = 0


class ConfigManager:
    """Handles configuration loading and management."""
    
    @staticmethod
    def load_config(config_path: str) -> FullAtlasConfig:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        return FullAtlasConfig(
            api_key=config['openai']['api_key'],
            base_url=config['openai']['base_url'],
            model_name=config['openai']['chat_model'],
            lkg_dir=config['kg2kv']['lkg_dir'],
            lkg_file=config['kg2kv']['lkg_file'],
            output_folder=config['kg2kv']['output_folder'],
            output_file=config['kg2kv']['output_folder'] + config['kg2kv']['output_file'],
            if_check=config['kg2kv']['if_check'],
            if_naive=config['kg2kv']['if_naive'],
            max_name_tokens=config['kg2kv']['max_name_tokens'],
            max_name_chars=config['kg2kv']['max_name_chars'],
            max_description_tokens=config['kg2kv']['max_description_tokens'],
            max_description_chars=config['kg2kv']['max_description_chars'],
        )


class SystemMonitor:
    """Handles system resource monitoring."""
    
    @staticmethod
    def check_memory() -> bool:
        """Check CPU and memory usage."""
        memory = psutil.virtual_memory()
        if memory.percent > 85:
            print(f"\033[91mWarning: Memory usage {memory.percent}%. Shutdown...\033[0m")
            return False
        return True


class TextProcessor:
    """Handles text processing operations."""
    
    @staticmethod
    def count_tokens(text: str) -> int:
        """Count tokens in text (simple word count)."""
        return len(text.split())

    @staticmethod
    def count_chars(text: str) -> int:
        """Count characters in text."""
        return len(text)


class RelationConverter:
    """Handles relation to natural language conversion."""
    
    def __init__(self, config: FullAtlasConfig):
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
                    {"role": "system", "content": self.relation_prompt_short},
                    {"role": "user", "content": f"relation: {relation}, missing: {missing}"}
                ]
            )
            return response.choices[0].message.content.strip()


class TripleValidator:
    """Handles triple validation and filtering."""
    
    def __init__(self, config: FullAtlasConfig):
        self.config = config
        self.text_processor = TextProcessor()

    def validate_triple_lengths(self, head: str, tail: str, missing: str) -> bool:
        """Validate triple lengths based on configuration."""
        if not self.config.if_check:
            return True
        
        if missing == 'head':
            if (self.text_processor.count_tokens(tail) >= self.config.max_name_tokens or 
                self.text_processor.count_chars(tail) >= self.config.max_name_chars):
                return False
            if (self.text_processor.count_tokens(head) >= self.config.max_description_tokens or 
                self.text_processor.count_chars(head) >= self.config.max_description_chars):
                return False
        else:
            if (self.text_processor.count_tokens(head) >= self.config.max_name_tokens or 
                self.text_processor.count_chars(head) >= self.config.max_name_chars):
                return False
            if (self.text_processor.count_tokens(tail) >= self.config.max_description_tokens or 
                self.text_processor.count_chars(tail) >= self.config.max_description_chars):
                return False
        return True


class BatchManager:
    """Handles batch file operations."""
    
    def __init__(self, config: FullAtlasConfig):
        self.config = config
        self.validator = TripleValidator(config)
        self.monitor = SystemMonitor()

    def save_raw_triples_batches(self, max_triples: int = 150000, batch_size: int = 10000) -> int:
        """Save raw triples in batches to files."""
        print(f"=== Step 1: Saving raw triple batches ===")
        print(f"Target count: {max_triples}, Batch size: {batch_size}")
        
        os.makedirs(self.config.output_folder, exist_ok=True)
        
        batch_num = 0
        total_processed = 0
        triples_batch = []
        
        with open(f"{self.config.lkg_dir}/{self.config.lkg_file}", 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            for row in reader:
                if total_processed >= max_triples:
                    break
                
                if not self.monitor.check_memory():
                    print("Insufficient memory, stopping processing")
                    break
                
                head, tail, relation = row[0], row[1], row[2]
                missing = random.choice(['head', 'tail'])
                
                if not self.validator.validate_triple_lengths(head, tail, missing):
                    continue
                
                triples_batch.append({'head': head, 'tail': tail, 'relation': relation, 'missing': missing})
                print(f"Processed triples: {total_processed}", end="\r")
                total_processed += 1
                
                if len(triples_batch) >= batch_size:
                    self._save_batch_file(triples_batch, batch_num)
                    print(f"Saved raw batch {batch_num}, containing {len(triples_batch)} triples")
                    batch_num += 1
                    triples_batch = []
            
            if triples_batch:
                self._save_batch_file(triples_batch, batch_num)
                print(f"Saved raw batch {batch_num}, containing {len(triples_batch)} triples")
                batch_num += 1
        
        print(f"Raw triple batches saved! Total {batch_num} batches, {total_processed} triples")
        return batch_num

    def _save_batch_file(self, triples_batch: List[Dict], batch_num: int) -> None:
        """Save a batch of triples to file."""
        raw_batch_file = os.path.join(self.config.output_folder, f"raw_batch_{batch_num:06d}.json")
        with open(raw_batch_file, 'w', encoding='utf-8') as f:
            json.dump(triples_batch, f, ensure_ascii=False, indent=2)

    def get_raw_batch_files(self) -> List[str]:
        """Get all raw batch files."""
        pattern = os.path.join(self.config.output_folder, "raw_batch_*.json")
        return sorted(glob.glob(pattern))

    def get_processed_batch_files(self) -> List[str]:
        """Get all processed batch files."""
        pattern = os.path.join(self.config.output_folder, "processed_batch_*.json")
        return sorted(glob.glob(pattern))


class DataProcessor:
    """Handles data processing and transformation."""
    
    def __init__(self, config: FullAtlasConfig):
        self.config = config
        self.relation_converter = RelationConverter(config)
        self.monitor = SystemMonitor()

    async def process_single_batch(self, raw_batch_file: str, batch_num: int, max_concurrent: int = 50) -> bool:
        """Process a single raw batch file."""
        print(f"Starting to process batch {batch_num}: {raw_batch_file}")
        
        if not self.monitor.check_memory():
            print(f"Insufficient memory, skipping batch {batch_num}")
            return False

        with open(raw_batch_file, 'r', encoding='utf-8') as f:
            triples_batch = json.load(f)
        
        print(f"Batch {batch_num} contains {len(triples_batch)} triples")
        
        if not self.config.if_naive:
            semaphore = asyncio.Semaphore(max_concurrent)
            async with AsyncOpenAI(api_key=self.config.api_key, base_url=self.config.base_url) as client:
                tasks = [self.relation_converter.convert_relation_to_nl(client, semaphore, t['relation'], t['missing']) for t in triples_batch]
                relation_nls = await tqdm.gather(*tasks, desc=f"Batch {batch_num} - Generating NL relations")
        else:
            relation_nls = [t['relation'] for t in triples_batch]

        results = self._process_batch_results(triples_batch, relation_nls)
        self._save_processed_batch(results, batch_num)
        
        del triples_batch, results, relation_nls
        return True

    def _process_batch_results(self, triples_batch: List[Dict], relation_nls: List[str]) -> List[Dict]:
        """Process batch results into final format."""
        results = []
        for i, triple_info in enumerate(triples_batch):
            head = triple_info['head']
            tail = triple_info['tail']
            missing = triple_info['missing']
            relation_nl = relation_nls[i].replace('"', '')

            name = f"{tail}" if missing == 'head' else f"{head}"
            key_string = f"the {relation_nl} of {name}"
            question = f"What is {key_string}?"
            answer = f"{key_string} is {head}" if missing == 'head' else f"{key_string} is {tail}"
            description_type = f"{relation_nl}"
            description = f"{head}" if missing == 'head' else f"{tail}"
            
            item = {
                "name": name,
                "description_type": description_type,
                "description": description,
                "Q": question,
                "A": answer,
                "key_string": key_string,
                "extended_Q": "",
                "extended_A": ""
            }
            results.append(item)
        return results

    def _save_processed_batch(self, results: List[Dict], batch_num: int) -> None:
        """Save processed batch results."""
        processed_batch_file = os.path.join(self.config.output_folder, f"processed_batch_{batch_num:06d}.json")
        with open(processed_batch_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Batch {batch_num} processing completed, saved to {processed_batch_file}")

    async def process_all_batches(self) -> int:
        """Process all raw batch files."""
        print(f"=== Step 2: Processing all raw batch files ===")
        
        batch_manager = BatchManager(self.config)
        raw_batch_files = batch_manager.get_raw_batch_files()
        if not raw_batch_files:
            print("No raw batch files found")
            return 0
        
        print(f"Found {len(raw_batch_files)} raw batch files")
        
        processed_count = 0
        for i, raw_batch_file in enumerate(raw_batch_files):
            batch_num = i
            success = await self.process_single_batch(raw_batch_file, batch_num)
            if success:
                processed_count += 1
            
            if not self.monitor.check_memory():
                print("Insufficient memory, stopping processing")
                break
        
        print(f"Batch processing completed! Successfully processed {processed_count} batches")
        return processed_count


class Merger:
    """Handles merging of processed batch files."""
    
    def __init__(self, config: FullAtlasConfig):
        self.config = config
        self.batch_manager = BatchManager(config)
        self.monitor = SystemMonitor()

    def merge_with_command(self) -> bool:
        """Merge processed batch files using command line tools."""
        print(f"=== Step 3: Merging processed batch files ===")
        
        processed_batch_files = self.batch_manager.get_processed_batch_files()
        if not processed_batch_files:
            print("No processed batch files found")
            return False
        
        print(f"Found {len(processed_batch_files)} processed batch files")
        
        output_file_jsonl = self.config.output_file.replace('.json', '.jsonl')
        cmd = f"jq -c '.[]' <(jq -s 'flatten' {' '.join(processed_batch_files)}) > {output_file_jsonl}"
        print(f"Trying jq command: {cmd}")
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"jq merge completed! Results saved to {output_file_jsonl}")
                return True
            else:
                print(f"jq merge failed: {result.stderr}")
                print("Trying Python method to merge...")
                return self.merge_with_python()
        except Exception as e:
            print(f"Error executing jq command: {e}")
            print("Trying Python method to merge...")
            return self.merge_with_python()

    def merge_with_python(self) -> bool:
        """Fallback method: Use Python to merge all processed batch files."""
        print("Using Python streaming merge method...")
        
        processed_batch_files = self.batch_manager.get_processed_batch_files()
        if not processed_batch_files:
            print("No processed batch files found")
            return False
        
        return self._streaming_merge(processed_batch_files)

    def _streaming_merge(self, processed_batch_files: List[str]) -> bool:
        """Streaming merge method for efficient memory usage."""
        print("Using streaming merge method...")
        
        total_records = self._count_total_records(processed_batch_files)
        print(f"Total records to merge: {total_records}")
        
        with open(self.config.output_file, 'w', encoding='utf-8') as output_f:
            records_written = 0
            random.shuffle(processed_batch_files)
            
            for batch_file in normal_tqdm(processed_batch_files, desc="Streaming merge to JSONL"):
                try:
                    with open(batch_file, 'r', encoding='utf-8') as f:
                        batch_data = json.load(f)
                        
                    random.shuffle(batch_data)
                    
                    for item in batch_data:
                        output_f.write(json.dumps(item, ensure_ascii=False) + '\n')
                        records_written += 1
                        
                        if records_written % 1000 == 0:
                            if not self.monitor.check_memory():
                                print("Insufficient memory during merge, stopping...")
                                return False
                                
                except Exception as e:
                    print(f"Error: Cannot read batch file {batch_file}: {e}")
                    continue
        
        print(f"Streaming merge completed! Total {records_written} records saved to {self.config.output_file}")
        return True

    def _count_total_records(self, processed_batch_files: List[str]) -> int:
        """Count total records across all batch files."""
        total_records = 0
        for batch_file in processed_batch_files:
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)
                    total_records += len(batch_data)
            except Exception as e:
                print(f"Error: Cannot read batch file {batch_file}: {e}")
                continue
        return total_records


class FullAtlasBuilder:
    """Main orchestrator for full Atlas dataset generation."""
    
    def __init__(self, config: FullAtlasConfig):
        self.config = config
        self.batch_manager = BatchManager(config)
        self.data_processor = DataProcessor(config)
        self.merger = Merger(config)

    async def build_dataset(self, max_triples: int = 100000, batch_size: int = 10000) -> None:
        """Build the complete Atlas dataset using three-step pipeline."""
        print("\033[94m[Starting three-step streaming processing for large dataset]\033[0m")
        
        # Step 1: Save raw triple batches
        print("\033[94m[Step 1: Saving raw triple batches]\033[0m")
        batch_count = self.batch_manager.save_raw_triples_batches(max_triples, batch_size)
        
        if batch_count > 0:
            # Step 2: Process all raw batch files
            print("\033[94m[Step 2: Processing all raw batch files]\033[0m")
            processed_count = await self.data_processor.process_all_batches()
            
            if processed_count > 0:
                # Step 3: Merge processed batch files
                print("\033[94m[Step 3: Merging processed batch files]\033[0m")
                success = self.merger.merge_with_command()
                
                if success:
                    print("\033[92m[Three-step streaming processing completed!]\033[0m")
                else:
                    print("\033[91m[Merge step failed]\033[0m")
            else:
                print("\033[91m[Batch processing failed]\033[0m")
        else:
            print("\033[91m[Raw batch saving failed]\033[0m")


def load_config_from_file(config_path: str) -> FullAtlasConfig:
    """Load configuration from YAML file."""
    return ConfigManager.load_config(config_path)


async def main():
    config_path = '../config/full_atlas_config.yaml'
    config = load_config_from_file(config_path)
    
    builder = FullAtlasBuilder(config)
    await builder.build_dataset(max_triples=100000, batch_size=10000)


if __name__ == "__main__":
    asyncio.run(main())
