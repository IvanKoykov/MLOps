from config.config import AppConfig
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import os
import shutil
from sklearn.model_selection import train_test_split


def move_file(file):
    target_dir = os.path.dirname(os.path.dirname(file))

    shutil.move(str(file), target_dir)


def split_data(full_path, dict_with_name_files):
    for path in full_path:
        for key, val in dict_with_name_files.items():
            if Path(path).parts[-1] == "mask_people_2":
                val = list(map(lambda x: x.replace("jpg", "png"), val))
            else:
                val = val
            test_path = os.path.join(path, key)
            list(
                map(lambda x: shutil.move(os.path.join(path, x), test_path), val)  # noqa: E501
            )


def split_data_on_train_val_test(full_path):
    train_dir = []
    val_dir = []
    test_dir = []
    for x in full_path:
        train_dir.append(os.path.join(x, "train"))
        val_dir.append(os.path.join(x, "val"))
        test_dir.append(os.path.join(x, "test"))
    train_size = 0.8
    for i in range(len(train_dir)):
        if not os.path.exists(train_dir[i]):
            os.mkdir(train_dir[i])
        if not os.path.exists(val_dir[i]):
            os.mkdir(val_dir[i])
        if not os.path.exists(test_dir[i]):
            os.mkdir(test_dir[i])

    files = [
        name
        for name in os.listdir(full_path[0])
        if os.path.isfile(os.path.join(full_path[0], name))
    ]  # собственно, я получил адреса всех файлов
    dict_with_name_files = {}
    (
        dict_with_name_files["train"],
        dict_with_name_files["test"],
    ) = train_test_split(  # noqa: E501
        files, train_size=train_size
    )
    (
        dict_with_name_files["val"],
        dict_with_name_files["test"],
    ) = train_test_split(  # noqa: E501
        dict_with_name_files["test"], test_size=0.5
    )
    split_data(full_path, dict_with_name_files)


def transport_data(config: AppConfig):
    images_path_list = [x for x in config.dataset_output_path.glob("**/*.*")]
    list(map(move_file, images_path_list))
    for image_path in images_path_list:
        del_path = os.path.dirname(image_path)
        if not os.path.exists(del_path):
            continue
        else:
            shutil.rmtree(del_path)


def main_actions(config: AppConfig):

    images_path_list = [x for x in config.dataset_path.glob("**/*.*")]
    dataset_path = config.dataset_output_path
    dataset_path.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(images_path_list, desc="Images saving"):

        class_name = image_path.parts[-2]  # Folder Name
        stage_name = image_path.parts[-3]  # Train/Test/Val
        class_folder = dataset_path / stage_name / class_name
        class_folder.mkdir(parents=True, exist_ok=True)
        Image.open(image_path).resize(size=(512, 512)).save(
            class_folder / image_path.name
        )

    transport_data(config)

    dirs = os.listdir(Path(config.dataset_output_path))
    full_path = [
        os.path.join(Path(config.dataset_output_path), x) for x in dirs
    ]

    split_data_on_train_val_test(full_path)


def main():
    config = AppConfig.parse_raw()
    main_actions(config=config)


if __name__ == "__main__":
    main()
