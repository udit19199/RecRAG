import json
import logging
import os
from pathlib import Path

import pandas as pd
from config import find_config_path, get_storage_dir, load_config
from datasets import Dataset
from evaluation.ragas_eval import get_evaluator
from pipelines import get_retrieval_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_batch_evaluation(
    dataset_path: str = "data/eval_dataset.json",
    output_dir: str = "storage",
):
    """Run RAGAS evaluation on a dataset of questions."""
    config_path = find_config_path()
    config = load_config(config_path)
    pipeline = get_retrieval_pipeline(config_path)
    evaluator = get_evaluator()

    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        logger.error(f"Dataset file {dataset_path} not found.")
        return

    with open(dataset_file, "r") as f:
        eval_data = json.load(f)

    logger.info(f"Evaluating {len(eval_data)} queries...")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in eval_data:
        question = item.get("question")
        ground_truth = item.get("ground_truth")

        if not question:
            continue

        logger.info(f"Processing query: {question[:50]}...")
        result = pipeline.query(question)

        questions.append(question)
        answers.append(result["response"])
        contexts.append([doc.text for doc in result["context"]])
        if ground_truth:
            ground_truths.append(ground_truth)

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    if ground_truths:
        data["ground_truth"] = ground_truths

    dataset = Dataset.from_dict(data)

    logger.info("Running RAGAS evaluation...")
    result = evaluator.evaluate_query_batch(dataset)

    # Serialize results
    output_path = get_storage_dir(config, config_path) / "evaluation_results.json"
    
    # result is an EvaluationResult object in RAGAS 0.1+
    # We can convert it to a pandas DataFrame and then to a list of dicts
    results_df = result.to_pandas()
    detailed_results = results_df.to_dict(orient="records")
    
    summary = {
        metric: results_df[metric].mean() 
        for metric in result.scores[0].keys()
    }

    final_output = {
        "summary": summary,
        "detailed_results": detailed_results
    }

    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=4)

    logger.info(f"Evaluation results saved to {output_path}")
    logger.info(f"Summary: {summary}")


if __name__ == "__main__":
    # Ensure RagasEvaluator has the method we need
    # I'll update the evaluator class to support batching
    run_batch_evaluation()
