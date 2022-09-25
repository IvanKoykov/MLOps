from config.config import AppConfig
from clearml import Task, TaskTypes, Dataset
from src.train import main_actions
from src.utils import split_path_train_test_val


def main():

    # id, version = Dataset._get_dataset_id(
    # dataset_project='segmentation_people'
    # ,dataset_name="people")
    config: AppConfig = AppConfig.parse_raw()
    # config.dataset_id = id
    # print(config.dataset_version,"   VERSION ")

    task: Task = Task.init(
        project_name="segmentation_people",
        task_name="training",
        task_type=TaskTypes.training,
    )

    # clearml_params = {
    #   "dataset_id": config.dataset_id
    # }
    clearml_params = config.dict()
    task.connect(clearml_params)
    dataset_path = Dataset.get(clearml_params["dataset_id"]).get_local_copy()
    # dataset_path='D:/ClearML/cache/storage_manager/datasets/ds_d83322ee415d49ce99f7fb0f92872a9b/'
    people_path, mask_path = split_path_train_test_val(dataset_path)
    # config: AppConfig = AppConfig.parse_raw()
    config.people_dataset_path = people_path
    config.mask_dataset_path = mask_path
    # print(config.people_dataset_path['val'],"  TASK")
    # print(config.dataset_id,"   DATASET_ID_TRAIN_TASK")
    main_actions(config=config)


if __name__ == "__main__":
    main()
