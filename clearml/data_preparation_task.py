from pathlib import Path
from clearml import Task, TaskTypes, Dataset
from config.config import AppConfig
from src.data_preparation import main_actions


def main():
    config: AppConfig = AppConfig.parse_raw()
    task: Task = Task.init(
        project_name="segmentation_people",
        task_name="data_preparation",
        task_type=TaskTypes.data_processing,
    )
    #clearml_params = {
     #   "dataset_id": '617f60ea548943149a09b737acb9d674'
     #}
    clearml_params = {"dataset_id": config.dataset_id}
    task.connect(clearml_params)
    dataset_path = Dataset.get(**clearml_params).get_local_copy()

    config.dataset_path = Path(dataset_path)
    main_actions(config=config)
    dataset = Dataset.create(
        dataset_project="segmentation_people", dataset_name="people"
    )
    dataset.add_files(config.dataset_output_path)
    task.set_parameter("output_dataset_id", dataset.id)
    dataset.upload()
    dataset.finalize()


if __name__ == "__main__":
    main()
