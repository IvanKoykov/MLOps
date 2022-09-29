from pathlib import Path
from clearml import Task, TaskTypes, Dataset
from config.config import AppConfig
from src.data_validation import main_actions


def main():
    id, version = Dataset._get_dataset_id(
        dataset_project="segmentation_people",
        dataset_name="people",
        dataset_version="1.0.0",
    )
    config: AppConfig = AppConfig.parse_raw()
    config.dataset_id = id
    # config.dataset_version=version
    print(config.dataset_id, " DATASET_ID_VALID_TASK")

    task: Task = Task.init(
        project_name="segmentation_people",
        task_name="data_validation",
        task_type=TaskTypes.data_processing,
    )
    clearml_params = {"dataset_id": config.dataset_id}
    task.connect(clearml_params)
    dataset_path = Dataset.get(**clearml_params).get_local_copy()
    config.dataset_path = Path(dataset_path)
    main_actions(config=config)


if __name__ == "__main__":
    main()
