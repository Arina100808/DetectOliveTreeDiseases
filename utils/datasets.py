from roboflow import Roboflow
import shutil
import os

def download_roboflow_dataset(author, project_name, version=1, path="data", data_format="yolov8"):
    api_key = os.getenv("ROBOFLOW_API_KEY")
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

    return dataset, dataset_dir
