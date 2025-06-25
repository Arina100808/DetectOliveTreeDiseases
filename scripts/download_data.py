from roboflow import Roboflow
import argparse
import shutil
import os

def download_roboflow_dataset(author, project_name, api_key, version=1, path="data", data_format="yolov8"):
    # api_key = os.getenv("ROBOFLOW_API_KEY")
    rf = Roboflow(api_key=api_key)

    project = rf.workspace(author).project(project_name)
    version = project.version(version)

    dataset = version.download(data_format)
    dataset_name = dataset.name
    dataset_name = dataset_name.replace(" ", "_").lower()
    target_path = os.path.join(path, dataset_name)
    shutil.move(dataset.location, target_path)
    dataset.location = target_path
    dataset_dir = dataset.location

def main():
    parser = argparse.ArgumentParser(description="Download Roboflow dataset")
    parser.add_argument('--author', type=str, default="neuroniella", help='Roboflow dataset author')
    parser.add_argument('--project_name', type=str, default="olive-tree-diseases", help='Project name on Roboflow')
    parser.add_argument('--version', type=int, default=1, help='Dataset version number')
    parser.add_argument('--path', type=str, default="data", help='Directory to download the dataset to')
    parser.add_argument('--data_format', type=str, default="yolov8", help='Dataset format (e.g., yolov5, yolov8, coco, etc.)')
    parser.add_argument('--api_key', type=str, required=True, help='Roboflow API key')

    args = parser.parse_args()

    download_roboflow_dataset(
        author=args.author,
        project_name=args.project_name,
        api_key=args.api_key,
        version=args.version,
        path=args.path,
        data_format=args.data_format
    )

if __name__ == "__main__":
    main()
