import os
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
import torch
from src.create_dataset import CustomDataset
from src.utils import from_tensor_to_numpy, visualize, to_numpy_for_test_data
from config.config import AppConfig

"""
TO DO:

2)Разобраться с функциями: to_numpy_for_test_data и from_tensor_to_numpy

Теория:
1)разобраться с onnx
"""

"""
def blur(path_img,onnx_session,DEVICE):

    orig_img = cv2.cvtColor(cv2.imread(path_img), cv2.COLOR_BGR2RGB)
    frame_height, frame_width = orig_img.shape[:2]
    image = orig_img
    image = cv2.resize(image, (256, 256))
    image = np.rollaxis(image, 2, 0)
    image = torch.Tensor(image)
    image = image.to(DEVICE).unsqueeze(0)

    onnx_inputs = {onnx_session.get_inputs()[0].name:
    to_numpy_for_test_data(image)}
    onnx_output = onnx_session.run(None, onnx_inputs)
    pred_mask = onnx_output[0]
    print(pred_mask.shape)
    pred_mask = pred_mask.squeeze(0)

    pred_mask = np.transpose(pred_mask, (1, 2, 0))
    pred_mask = cv2.resize(pred_mask, (frame_width, frame_height))
    pred_mask = cv2.threshold(pred_mask, 0.5, 1, cv2.THRESH_BINARY)[1]
    pred_mask = pred_mask.astype(np.uint8)

    output_image = cv2.GaussianBlur(orig_img, (-1, -1), 20)
    output_image[pred_mask == 1] = orig_img[pred_mask == 1]

    plt.imshow(pred_mask)
    plt.show()
    plt.imshow(orig_img)
    plt.show()
    plt.imshow(output_image)
    plt.show()
"""


def main_actions(config: AppConfig):
    x_test_dir = str(config.people_dataset_path["test"])
    y_test_dir = str(config.mask_dataset_path["test"])

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load weights

    if os.path.exists("path_to_file_with_weights.pth"):
        best_model = torch.load(  # noqa: F841
            "path_to_file_with_weights.pth", map_location=DEVICE
        )
        print("Loaded DeepLabV3+ model from this run.")

    # load from onnx
    onnx_session = ort.InferenceSession("Deep_Lab.onnx")
    test_dataset = CustomDataset(x_test_dir, y_test_dir)

    for idx in tqdm(range(len(test_dataset)), desc="Images testing"):
        image, gt_mask = test_dataset[idx]
        image, gt_mask = image.to(DEVICE).unsqueeze(0), gt_mask.to(DEVICE)  # noqa: E501
        onnx_inputs = {
            onnx_session.get_inputs()[0].name: to_numpy_for_test_data(
                image, DEVICE
            )  # noqa: E501
        }
        onnx_output = onnx_session.run(None, onnx_inputs)

        pred_mask = onnx_output[0]
        pred_mask = pred_mask.squeeze(0)
        pred_mask = np.transpose(pred_mask, (1, 2, 0))

        image = image.squeeze(0)
        image, gt_mask = from_tensor_to_numpy(image, gt_mask)

        visualize(
            original_image=image,
            ground_truth_mask=gt_mask,
            predict_mask=pred_mask,
        )

    # path_img = "path_to_the_image_for_the_blurr.jpg"
    # blur(path_img,onnx_session,DEVICE)


def main():
    config = AppConfig.parse_raw()
    print("MAIN")
    main_actions(config=config)


if __name__ == "__main__":
    main()
