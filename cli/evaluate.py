import json
import logging
from pathlib import Path

from datasets import Dataset

from config import find_config_path, get_storage_dir, load_config
from evaluation.ragas_eval import get_evaluator
from pipelines import get_retrieval_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_batch_evaluation(
    dataset_path: str = "data/eval_dataset.json",
    output_dir: str = "storage",
) -> None:
    """Run RAGAS evaluation on a dataset of questions.

    Args:
        dataset_path: Path to JSON file containing evaluation questions.
        output_dir: Directory to write evaluation_results.json.
    """
    config_path = find_config_path()
    config = load_config(config_path)
    pipeline = get_retrieval_pipeline(config_path)
    evaluator = get_evaluator()

    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        logger.error("Dataset file %s not found.", dataset_path)
        return

    with open(dataset_file) as f:
        eval_data = json.load(f)

    logger.info("Evaluating %d queries...", len(eval_data))

    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    for item in eval_data:
        question = item.get("question")
        ground_truth = item.get("ground_truth")

        if not question:
            continue

        logger.info("Processing query: %s...", question[:50])
        result = pipeline.query(question)

        questions.append(question)
        answers.append(result["response"])
        contexts.append([doc.text for doc in result["context"]])
        if ground_truth:
            ground_truths.append(ground_truth)

    data: dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    if ground_truths:
        data["ground_truth"] = ground_truths

    dataset = Dataset.from_dict(data)

    logger.info("Running RAGAS evaluation...")
    result = evaluator.evaluate_query_batch(dataset)

    output_path = get_storage_dir(config, config_path) / "evaluation_results.json"

    results_df = result.to_pandas()
    detailed_results = results_df.to_dict(orient="records")

    summary = {metric: results_df[metric].mean() for metric in result.scores[0].keys()}

    final_output = {
        "summary": summary,
        "detailed_results": detailed_results,
    }

    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=4)

    logger.info("Evaluation results saved to %s", output_path)
    logger.info("Summary: %s", summary)


if __name__ == "__main__":
    run_batch_evaluation()
