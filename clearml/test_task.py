from config.config import AppConfig
from clearml import Task, TaskTypes, Dataset
from src.test import main_actions
from src.utils import split_path_train_test_val


def main():

    config: AppConfig = AppConfig.parse_raw()

    task: Task = Task.init(
        project_name="segmentation_people",
        task_name="testing",
        task_type=TaskTypes.testing,
    )

    clearml_params = config.dict()
    print(clearml_params, "    PARAMS")
    # clearml_params = {
    #   "dataset_id": 'd33d5cc79120482e9892e25cecb78972'}

    task.connect(clearml_params)
    dataset_path = Dataset.get(clearml_params["dataset_id"]).get_local_copy()
    people_path, mask_path = split_path_train_test_val(dataset_path)
    config.people_dataset_path = people_path
    config.mask_dataset_path = mask_path
    print(config.dataset_id, "   DATASET_ID_TEST_TASK")
    main_actions(config=config)


if __name__ == "__main__":
    main()
