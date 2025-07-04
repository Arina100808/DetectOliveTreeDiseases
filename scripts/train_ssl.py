from lightly.transforms import DINOTransform
import lightly_train
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default="experiments/yolov8_ssl",
                        help='Output directory for SSL training results')
    parser.add_argument('--data', type=str, default="data/unlabeled_olive_trees",
                        help='Directory with unlabeled images')
    parser.add_argument('--model', type=str, default="ultralytics/best.yaml",
                        help='YOLO model YAML to use (e.g., ultralytics/best.yaml)')
    parser.add_argument('--method', type=str, default="distillation",
                        help='SSL method to use (e.g., distillation, simclr, etc.)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for training')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a checkpoint to resume training')
    parser.add_argument('--resume_interrupted', action='store_true',
                        help='Resume training from the last checkpoint if interrupted')

    args = parser.parse_args()

    lightly_train.train(
        out=args.out,
        data=args.data,
        model=args.model,
        method=args.method,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint=args.checkpoint,
        resume_interrupted=args.resume_interrupted,
        callbacks={"model_checkpoint": {"every_n_epochs": 10}, "model_export": {"every_n_epochs": 10}},
    )


if __name__ == "__main__":
    main()
