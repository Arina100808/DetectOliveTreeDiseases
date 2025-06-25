from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='best.pt', help='YOLO model')
    parser.add_argument('--data', type=str, default='data/olive_tree_diseases/data.yaml',
                        help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr0', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--momentum', type=float, default=0.937, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='SGD weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='Number of warmup epochs')
    parser.add_argument('--cos_lr', action='store_true', help='Use cosine learning rate scheduler')
    parser.add_argument('--close_mosaic', type=int, default=10, help='Close mosaic augmentation')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--plots', action='store_true', help='Save training plots')
    parser.add_argument('--freeze', type=int, nargs='+', default=[None], help='Freeze model layers')
    parser.add_argument('--project', type=str, default='experiments/yolov8n_baseline')
    parser.add_argument('--name', type=str, default=None)

    args = parser.parse_args()

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        optimizer=args.optimizer,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic,
        patience=args.patience,
        plots=args.plots,
        freeze=args.freeze,
        project=args.project,
        name=args.name
    )

if __name__ == "__main__":
    main()
