from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov8n_best', help='Model name')
    parser.add_argument('--data', type=str, default='data/olive_tree_diseases/data.yaml',
                        help='Path to data.yaml')
    parser.add_argument('--path', type=str, default='models')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to evaluate on')
    parser.add_argument('--project', type=str, default='evaluation_results', help='Project directory for saving results')

    args = parser.parse_args()

    model = YOLO(f'{args.path}/{args.model}.pt')  # Trained weights
    metrics = model.val(data=args.data, split=args.split, project='evaluation_results', name=f'{args.model}_eval')

    print(f"\nResults for {args.model} model:")
    print(f"mAP50: {metrics.results_dict['metrics/mAP50(B)']}, mAP50-95: {metrics.results_dict['metrics/mAP50-95(B)']}")
    print(f"precision: {metrics.results_dict['metrics/precision(B)']}, recall: {metrics.results_dict['metrics/recall(B)']}")


if __name__ == "__main__":
    main()