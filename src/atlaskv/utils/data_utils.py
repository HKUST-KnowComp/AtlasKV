import json
from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np


@dataclass
class Entity:
    """Entity data structure."""
    name: str
    description: str
    objectives: str
    purpose: str


@dataclass
class DataPoint:
    """Data point structure for QA pairs."""
    name: str
    description_type: str
    description: str
    Q: str = None
    A: str = None
    key_string: str = None
    extended_Q: str = None
    extended_A: str = None


class FileManager:
    """Handles file operations for entities and data points."""
    
    @staticmethod
    def save_entity(entity: Union[Entity, DataPoint], output_file: str) -> None:
        """Save entity to file in JSONL format."""
        try:
            with open(output_file, "a+") as f:
                json.dump(entity.__dict__, f)
                f.write("\n")
        except Exception as e:
            print("Error saving entity.")
            print(e)

    @staticmethod
    def load_entities(file_path: str) -> List[Union[Entity, DataPoint]]:
        """Load entities from JSONL file."""
        entities = []
        try:
            with open(file_path, "r") as f:
                for line in f:
                    entity_data = json.loads(line)
                    entities.append(entity_data)
        except Exception as e:
            print("Error loading entities.")
            print(e)
        return entities


class QuestionGenerator:
    """Handles question generation and augmentation."""
    
    QUESTION_TEMPLATES = [
        "What {} does {} have?",
        "What is the {} of {}?",
        "Tell me about the {} of {}.",
        "Can you let me know the {} of {}?",
        "Can you inform me about the {} of {}?",
        "Describe the {} of {}.",
        "What details can you share about the {} of {}?",
        "What kind of {} does {} have?",
        "Provide details on the {} of {}.",
        "What features does the {} of {} include?",
        "Can you elaborate on the {} of {}?",
        "How would you describe the {} of {}?",
        "What can you tell me about the {} characteristics of {}?",
        "Can you explain the {} of {}?",
        "What insights can you provide about the {} of {}?",
        "What should I know about the {} of {}?",
    ]

    MULTI_ENTITY_TEMPLATES = [
        "What is {}?",
        "Tell me {}.",
        "Can you let me know {}?",
        "Can you inform me {}?",
        "Describe {}.",
        "Explain {}.",
        "Could you describe the {}?",
        "What can you tell me about {}?",
        "Could you provide information on {}?",
        "Please enlighten me about {}.",
        "Can you clarify {} for me?",
        "Could you give me a detailed description of {}?",
        "I need more information on {}.",
    ]

    @classmethod
    def augment_entity(cls, row: dict) -> str:
        """Generate question for entity using random template."""
        dtype = row["description_type"]
        name = row["name"]
        template_idx = np.random.randint(0, len(cls.QUESTION_TEMPLATES))
        return cls.QUESTION_TEMPLATES[template_idx].format(dtype, name)

    @classmethod
    def generate_multi_entity_qa(cls, names: List[str], properties: List[str], answers: List[str]) -> Tuple[str, str]:
        """Generate question-answer pair for multiple entities."""
        template_idx = np.random.randint(0, len(cls.MULTI_ENTITY_TEMPLATES))
        
        # Build question body
        question_parts = []
        for name, property in zip(names[:-1], properties[:-1]):
            question_parts.append(f"the {property} of {name}")
        question_parts.append(f"the {properties[-1]} of {names[-1]}")
        question_body = ", ".join(question_parts)
        
        # Build answer string
        answer_parts = []
        for answer, name, property in zip(answers, names, properties):
            answer_parts.append(f"The {property} of {name} is {answer}")
        answer_str = "; ".join(answer_parts)
        
        return cls.MULTI_ENTITY_TEMPLATES[template_idx].format(question_body), answer_str


class ResponseHandler:
    """Handles response generation and fallback answers."""
    
    @staticmethod
    def get_unknown_response() -> str:
        """Get standard response for unknown queries."""
        return "I am sorry I cannot find relevant information in the KB."


# Backward compatibility functions
def save_entity(pair: Union[Entity, DataPoint], output_file: str) -> None:
    """Save entity to file (backward compatibility)."""
    FileManager.save_entity(pair, output_file)


def load_entities(inout_file: str) -> List[Union[Entity, DataPoint]]:
    """Load entities from file (backward compatibility)."""
    return FileManager.load_entities(inout_file)


def get_i_dont_know_ans() -> str:
    """Get unknown response (backward compatibility)."""
    return ResponseHandler.get_unknown_response()


def augment_row(row: dict) -> str:
    """Augment entity with question (backward compatibility)."""
    return QuestionGenerator.augment_entity(row)


def generate_multi_entity_qa(names: List[str], properties: List[str], answers: List[str]) -> Tuple[str, str]:
    """Generate multi-entity QA (backward compatibility)."""
    return QuestionGenerator.generate_multi_entity_qa(names, properties, answers)
