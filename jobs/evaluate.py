import json
import logging
from pathlib import Path

from config import find_config_path, load_config, get_storage_dir
from evaluation.ragas_eval import get_evaluator
from pipelines import get_retrieval_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_batch_evaluation(
    dataset_path: str = "data/eval_dataset.json",
) -> None:
    """Run RAGAS evaluation on a dataset of questions.

    Args:
        dataset_path: Path to JSON file containing evaluation questions.
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
    contexts_list: list[list[str]] = []
    ground_truths: list[str | None] = []

    for item in eval_data:
        question = item.get("question")
        ground_truth = item.get("ground_truth")

        if not question:
            continue

        logger.info("Processing query: %s...", question[:50])
        result = pipeline.query(question)

        questions.append(question)
        answers.append(result["response"])
        contexts_list.append([doc.text for doc in result["context"]])
        ground_truths.append(ground_truth or None)

    if not questions:
        logger.warning("No valid questions found in dataset.")
        return

    logger.info("Running RAGAS evaluation...")
    scores_list = evaluator.evaluate_query_batch(
        questions,
        contexts_list,
        answers,
        ground_truths=ground_truths,
    )

    output_path = get_storage_dir(config, config_path) / "evaluation_results.json"

    # Compute summary across all queries
    if scores_list:
        all_metrics = list(scores_list[0].keys())
        summary = {}
        for metric in all_metrics:
            values = [
                s[metric] for s in scores_list if metric in s and s[metric] is not None
            ]
            summary[metric] = sum(values) / len(values) if values else 0.0
    else:
        summary = {}

    final_output = {
        "summary": summary,
        "detailed_results": [
            {"question": q, **s} for q, s in zip(questions, scores_list)
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=4)

    logger.info("Evaluation results saved to %s", output_path)
    logger.info("Summary: %s", summary)


if __name__ == "__main__":
    run_batch_evaluation()
