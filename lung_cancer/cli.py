"""Command-line entry point for training, inference, evaluation, and serving."""

import argparse
import json
import logging
from pathlib import Path

from lung_cancer.config import get_settings
from lung_cancer.evaluation import evaluate
from lung_cancer.inference import InferenceService
from lung_cancer.logging_config import configure_logging
from lung_cancer.preprocessing import load_and_preprocess
from lung_cancer.reports import create_report
from lung_cancer.training import train
from lung_cancer.visualization import plot_confusion_matrix, plot_training_history


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lung-cancer", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="train the CNN from a labeled dataset")
    training.add_argument("--data", required=True, help="Dataset root")
    training.add_argument("--manifest", help="Optional CSV with image_path,label[,patient_id]")
    training.add_argument("--output", default="artifacts")
    training.add_argument("--epochs", type=int, default=50)
    training.add_argument("--batch-size", type=int, default=32)
    training.add_argument("--seed", type=int, default=get_settings().random_seed)
    prediction = commands.add_parser("predict", help="classify one image")
    prediction.add_argument("image")
    prediction.add_argument("--model", default=str(get_settings().model_path))
    evaluation = commands.add_parser("evaluate", help="evaluate on a labeled held-out set")
    evaluation.add_argument("--data", required=True)
    evaluation.add_argument("--manifest")
    evaluation.add_argument("--model", default=str(get_settings().model_path))
    evaluation.add_argument("--output", default="artifacts/evaluation.json")
    plot = commands.add_parser("plot", help="plot training or evaluation metrics")
    plot.add_argument("kind", choices=("training", "confusion"))
    plot.add_argument("--input", required=True)
    plot.add_argument("--output", required=True)
    serve = commands.add_parser("serve", help="start the HTTP API")
    serve.add_argument("--host", default=get_settings().host)
    serve.add_argument("--port", type=int, default=get_settings().port)
    return parser


def main() -> None:
    """Run the selected subcommand."""
    args = _parser().parse_args()
    configure_logging(get_settings().log_level)
    logger = logging.getLogger("lung_cancer")
    if args.command == "train":
        result = train(
            args.data, args.output, args.manifest, args.epochs, args.batch_size, args.seed
        )
    elif args.command == "predict":
        model_path = Path(args.model)
        service = InferenceService.from_checkpoint(model_path)
        image = load_and_preprocess(args.image)
        probabilities = service.model.predict(image[None, ...], verbose=0)[0]
        result = create_report(probabilities, model_path.stem).to_dict()
    elif args.command == "evaluate":
        result = evaluate(args.model, args.data, args.output, args.manifest)
    elif args.command == "plot":
        path = (
            plot_training_history(args.input, args.output)
            if args.kind == "training"
            else plot_confusion_matrix(args.input, args.output)
        )
        result = {"plot": str(path)}
    else:
        import uvicorn

        uvicorn.run("lung_cancer.api:app", host=args.host, port=args.port, reload=False)
        return
    logger.info("%s", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
