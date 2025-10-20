import argparse
import json
from dataclasses import dataclass
from typing import List

import numpy as np
from tqdm import tqdm
from atlaskv.gpt_session import GPT


@dataclass
class ScoredExample:
    text: str
    true_answer: str
    score: float


def append_jsonl(example: ScoredExample, output_file: str) -> None:
    try:
        with open(output_file, "a+") as f:
            json.dump(example.__dict__, f)
            f.write("\n")
    except Exception as e:  # noqa: BLE001
        print("Error saving example.")
        print(e)


class TextEvaluator(GPT):
    def __init__(self, model, endpoint_url, endpoint_api_key, **kwargs) -> None:
        self.system_msg = (
            "You are an AI system that evaluates the quality of generated text. "
            "You will be given a text and a ground truth answer, your goal is to return a score between 0 and 1."
        )
        self._template = (
            " Given a text and a ground truth answer, evaluate the quality of the text.\n"
            " Return a score of 1 if the text is exactly the same as the ground truth answer,\n"
            " Return a score of 0 if the text is completely wrong,\n"
            " Return a score between 0 and 1 if the text is partially correct. A more correct text should have a higher score.\n"
            " Do NOT generate anything else.\n"
            " Example:\n\n"
            ' Model output: "The sky is blue."\n'
            ' True answer: "The sky is blue."\n'
            " Score: 1\n\n"
            " Example 2:\n"
            ' Model output: "The color of Alexandria is blue."\n'
            ' True answer: "The color of Alexandria is green."\n'
            " Score: 0\n\n"
            " Example 3:\n"
            ' Model output: "The purpose of Alexandria is to extract knowledge."\n'
            ' True answer: "The color of Alexandria is to discover and organize knowledge into a structured form."\n'
            " Score: 0.9\n\n"
            " **Important**: Only generate a number.\n"
            "\n Score the following text: \n model prediction: {pred}, \n true answer: {ans}"
        )
        self.seed = 42
        super().__init__(model, endpoint_url=endpoint_url, endpoint_api_key=endpoint_api_key, **kwargs)

    def score_single(self, prediction: str, reference: str) -> ScoredExample:
        prompt = self._template.format(pred=prediction, ans=reference)
        score = self.generate_response(prompt)
        return ScoredExample(prediction, reference, float(score))

    def score_batch(self, lines: List[str]) -> List[ScoredExample]:
        results: List[ScoredExample] = []
        for raw in tqdm(lines):
            try:
                if "<|begin_of_text|>" in raw:
                    pred = (
                        raw.split("True answer:")[0]
                        .replace("True KB <|begin_of_text|> ", "")
                        .split("?")[1]
                        .strip()
                    )
                else:
                    pred = raw.split("True answer:")[0].replace("True KB:", "")
                ref = raw.split("True answer:")[1].strip()
                results.append(self.score_single(pred, ref))
            except Exception as e:  # noqa: BLE001
                print("Error evaluating example.")
                print(e)
        return results


def build_args():
    p = argparse.ArgumentParser(description="Text scoring with GPT")
    p.add_argument("--model", type=str, default="gpt-4o", help="The model to use.")
    p.add_argument("--endpoint_url", type=str, default="your_endpoint_url")
    p.add_argument("--endpoint_api_key", type=str, default="your_endpoint_api_key")
    p.add_argument(
        "--predictions_file",
        type=str,
        default="your_output_file_path/llama.txt",
        help="The input file with examples.",
    )
    p.add_argument(
        "--output_file",
        type=str,
        default="your_output_file_path/eval.json",
        help="The output file to save the examples.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = build_args()
    with open(args.predictions_file, "r") as f:
        content = f.read()
        lines = content.split("--------------------")

    evaluator = TextEvaluator(args.model, args.endpoint_url, args.endpoint_api_key)
    scored = evaluator.score_batch(lines)
    for ex in scored:
        append_jsonl(ex, args.output_file)

    mean_score = np.mean([ex.score for ex in scored])
    print(f"Mean score: {mean_score}")