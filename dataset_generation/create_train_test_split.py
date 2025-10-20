import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass
class SplitConfig:
    """Configuration for train-test split."""
    data_path: str = ""
    embedding_keys_path: str = ""
    embedding_values_path: str = ""
    output_path: str = ""
    split_index: int = 0


class FilePathManager:
    """Handles file path operations and naming."""
    
    @staticmethod
    def generate_split_names(original_path: str) -> Tuple[str, str]:
        """Generate train and test file names from original path."""
        base_name = Path(original_path).name
        train_name = f"train_{base_name}"
        test_name = f"test_{base_name}"
        return train_name, test_name

    @staticmethod
    def ensure_output_directory(output_path: str) -> Path:
        """Ensure output directory exists."""
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir


class DataLoader:
    """Handles data loading operations."""
    
    @staticmethod
    def load_json_data(file_path: str) -> List[dict]:
        """Load JSON data from file."""
        with open(file_path, "r") as f:
            return json.load(f)

    @staticmethod
    def load_embedding_data(file_path: str) -> np.ndarray:
        """Load embedding data from numpy file."""
        return np.load(file_path).astype("float32")


class DataSplitter:
    """Handles data splitting operations."""
    
    def __init__(self, config: SplitConfig):
        self.config = config
        self.path_manager = FilePathManager()
        self.data_loader = DataLoader()

    def split_json_data(self, data: List[dict], split_index: int) -> Tuple[List[dict], List[dict]]:
        """Split JSON data into train and test sets."""
        train_data = data[:split_index]
        test_data = data[split_index:]
        return train_data, test_data

    def split_embedding_data(self, embeddings: np.ndarray, split_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Split embedding data into train and test sets."""
        train_embeddings = embeddings[:split_index]
        test_embeddings = embeddings[split_index:]
        return train_embeddings, test_embeddings


class DataSaver:
    """Handles data saving operations."""
    
    @staticmethod
    def save_json_data(data: List[dict], file_path: Path) -> None:
        """Save JSON data to file."""
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def save_embedding_data(embeddings: np.ndarray, file_path: Path) -> None:
        """Save embedding data to numpy file."""
        np.save(file_path, embeddings)


class SplitProcessor:
    """Main processor for train-test split operations."""
    
    def __init__(self, config: SplitConfig):
        self.config = config
        self.path_manager = FilePathManager()
        self.data_loader = DataLoader()
        self.splitter = DataSplitter(config)
        self.saver = DataSaver()

    def load_all_data(self) -> Tuple[List[dict], np.ndarray, np.ndarray]:
        """Load all required data files."""
        json_data = self.data_loader.load_json_data(self.config.data_path)
        key_embeddings = self.data_loader.load_embedding_data(self.config.embedding_keys_path)
        value_embeddings = self.data_loader.load_embedding_data(self.config.embedding_values_path)
        return json_data, key_embeddings, value_embeddings

    def split_all_data(self, json_data: List[dict], key_embeddings: np.ndarray, value_embeddings: np.ndarray) -> Tuple[List[dict], List[dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split all data into train and test sets."""
        train_json, test_json = self.splitter.split_json_data(json_data, self.config.split_index)
        train_keys, test_keys = self.splitter.split_embedding_data(key_embeddings, self.config.split_index)
        train_values, test_values = self.splitter.split_embedding_data(value_embeddings, self.config.split_index)
        return train_json, test_json, train_keys, test_keys, train_values, test_values

    def generate_output_paths(self) -> Tuple[Path, Path, Path, Path, Path, Path]:
        """Generate all output file paths."""
        output_dir = self.path_manager.ensure_output_directory(self.config.output_path)
        
        train_data_name, test_data_name = self.path_manager.generate_split_names(self.config.data_path)
        train_keys_name, test_keys_name = self.path_manager.generate_split_names(self.config.embedding_keys_path)
        train_values_name, test_values_name = self.path_manager.generate_split_names(self.config.embedding_values_path)
        
        return (
            output_dir / train_data_name,
            output_dir / test_data_name,
            output_dir / train_keys_name,
            output_dir / test_keys_name,
            output_dir / train_values_name,
            output_dir / test_values_name
        )

    def save_all_splits(self, train_json: List[dict], test_json: List[dict], 
                       train_keys: np.ndarray, test_keys: np.ndarray,
                       train_values: np.ndarray, test_values: np.ndarray) -> None:
        """Save all split data to files."""
        (train_data_path, test_data_path, train_keys_path, test_keys_path, 
         train_values_path, test_values_path) = self.generate_output_paths()
        
        self.saver.save_json_data(train_json, train_data_path)
        self.saver.save_json_data(test_json, test_data_path)
        self.saver.save_embedding_data(train_keys, train_keys_path)
        self.saver.save_embedding_data(test_keys, test_keys_path)
        self.saver.save_embedding_data(train_values, train_values_path)
        self.saver.save_embedding_data(test_values, test_values_path)

    def process_split(self) -> None:
        """Execute the complete train-test split process."""
        # Load all data
        json_data, key_embeddings, value_embeddings = self.load_all_data()
        
        # Split all data
        train_json, test_json, train_keys, test_keys, train_values, test_values = self.split_all_data(
            json_data, key_embeddings, value_embeddings
        )
        
        # Save all splits
        self.save_all_splits(train_json, test_json, train_keys, test_keys, train_values, test_values)


def create_config_from_args(args: argparse.Namespace) -> SplitConfig:
    """Create configuration from command line arguments."""
    return SplitConfig(
        data_path=args.data_path,
        embedding_keys_path=args.embedding_keys_path,
        embedding_values_path=args.embedding_values_path,
        output_path=args.output_path,
        split_index=args.split_index,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the main data file to be split")
    parser.add_argument("--embedding_keys_path", type=str, help="Path to the file containing embedding keys")
    parser.add_argument("--embedding_values_path", type=str, help="Path to the file containing embedding values")
    parser.add_argument("--output_path", type=str, help="Directory path where the split datasets will be saved")
    parser.add_argument("--split_index", type=int, help="Index at which to split the data into train and test sets")
    return parser


def main():
    """Main function."""
    args = build_parser().parse_args()
    config = create_config_from_args(args)
    
    processor = SplitProcessor(config)
    processor.process_split()


if __name__ == "__main__":
    main()
