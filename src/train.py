import random as r
import cv2
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from src.create_dataset import CustomDataset
from src.utils import from_tensor_to_numpy, visualize
from src.model import model_DeepLabV3
from config.config import AppConfig


def main_actions(config: AppConfig):
    x_train_dir = str(config.people_dataset_path["train"])
    y_train_dir = str(config.mask_dataset_path["train"])
    x_val_dir = str(config.people_dataset_path["val"])
    y_val_dir = str(config.mask_dataset_path["val"])

    dataset = CustomDataset(x_train_dir, y_train_dir)
    random_idx = r.randint(0, len(dataset) - 1)
    image, mask = dataset[random_idx]

    image_arr, mask_arr = from_tensor_to_numpy(image, mask)

    visualize(
        original_image=image_arr,
        ground_truth_mask=(mask_arr),
        binar_mask=cv2.threshold(mask_arr, 0.5, 255, cv2.THRESH_BINARY)[1],
    )

    # create segmentation model with pretrained encoder
    ENCODER = "resnet50"
    ENCODER_WEIGHTS = "imagenet"
    ACTIVATION = "sigmoid"

    # preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS) # noqa: E501

    train_dataset = CustomDataset(x_train_dir, y_train_dir)
    valid_dataset = CustomDataset(x_val_dir, y_val_dir)

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=False, num_workers=2
    )  # noqa: E501
    valid_loader = DataLoader(
        valid_dataset, batch_size=1, shuffle=False, num_workers=2
    )  # noqa: E501

    model = model_DeepLabV3(ENCODER, ENCODER_WEIGHTS, ACTIVATION)

    # Set num of epochs
    EPOCHS = config.num_epochs

    # Set device: `cuda` or `cpu`
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define loss function
    loss = smp.utils.losses.DiceLoss()

    # define metrics
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    # define optimizer
    optimizer = torch.optim.Adam(
        [
            dict(params=model.parameters(), lr=config.lr),
        ]
    )

    # define learning rate scheduler (not used in this NB)
    """
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=1,
        T_mult=2,
        eta_min=5e-5,
    )
    """
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    train_model(
        model, train_epoch, valid_epoch, train_loader, valid_loader, EPOCHS
    )  # noqa: E501


def train_model(
    model, train_epoch, valid_epoch, train_loader, valid_loader, epochs
):  # noqa: E501

    # best_iou_score = 0.0
    train_logs_list, valid_logs_list = [], []

    for i in range(0, epochs):
        # Perform training & validation
        print("\nEpoch: {}".format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)

    # torch.save(model,
    # './DeepLab_model_1channel_mask_andrey_dataset.pth')  # save weights

    # save if onnx
    model.eval()
    # model.to("cpu")
    model.to(train_epoch.device)
    dummy_input = torch.randn(1, 3, 256, 256)
    input_names = ["actual_input"]
    output_names = ["output"]
    torch.onnx.export(
        model,
        dummy_input,
        "Deep_Lab_test.onnx",
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        do_constant_folding=True,
        opset_version=11,
    )


def main():
    config = AppConfig.parse_raw()
    print("MAIN")
    main_actions(config=config)


if __name__ == "__main__":
    main()

    # save_model(model)
