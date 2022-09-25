from typing import Union
from pathlib import Path
from pydantic_yaml import YamlModel


class AppConfig(YamlModel):
    # data
    dataset_path: Path
    dataset_output_path: Path
    training_dataset_path: Path
    valid_dataset_path: Path
    people_dataset_path: Path
    mask_dataset_path: Path
    dataset_id: str
    dataset_version: str

    # model training
    lr: float
    momentum: float
    num_epochs: int
    # validation
    val_acc_threshold: float
    val_loss_threshold: float

    @classmethod
    def parse_raw(cls, filename: Union[str, Path] = "D:/иван/MLOps/MLOps/config.yaml", *args, **kwargs):  # noqa: E501
        with open(filename, 'r') as f:
            data = f.read()
        return super().parse_raw(data, *args, **kwargs)

    # def __init__(self, *args, **kwargs) -> None:
    # super().__init__(*args, **kwargs)
