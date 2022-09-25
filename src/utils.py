import matplotlib.pyplot as plt
import albumentations as album
import torch
import os
from pathlib import Path


def from_tensor_to_numpy(image, mask):
    image_arr = image.cpu().detach().numpy()
    mask_arr = mask.cpu().detach().numpy()
    image_arr = image_arr.transpose(1, 2, 0).astype("int")
    mask_arr = mask_arr.transpose(1, 2, 0)
    return image_arr, mask_arr


def to_numpy_for_test_data(tensor, DEVICE):
    if str(DEVICE) == "cpu":
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )
    elif DEVICE == "cuda":
        return (
            tensor.detach().cuda().numpy()
            if tensor.requires_grad
            else tensor.cuda().numpy()
        )
    else:
        print("Процессор не определен")


def to_tensor(x, **kwargs):
    # print(x,'  ',kwargs)
    x = torch.Tensor(x)
    return x


def split_path_train_test_val(dataset_path):
    a = os.listdir(dataset_path)
    people = {}
    mask = {}
    for i in range(len(a)):
        temp = os.path.join(dataset_path, a[i])
        path = os.listdir(temp)
        for x in path:
            if i == 0:
                people[x] = Path(os.path.join(temp, x))
            elif i == 1:
                mask[x] = Path(os.path.join(temp, x))
            else:
                print("Неправильная структура директории с данными")
    return people, mask


def prerocessing(image, mask):
    _transform = []
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
    return album.Compose(_transform)


def visualize(**images):
    """
    Plot images in one row
    """
    # image_arr=image.cpu().detach().numpy()
    # mask_arr = mask.cpu().detach().numpy()
    # image_arr=image_arr.transpose(1,2,0).astype('int')
    # mask_arr=mask_arr.transpose(1,2,0)
    # print(image.shape,' ',type(image))

    n_images = len(images)
    plt.figure(figsize=(20, 8))
    for idx, (name, image) in enumerate(images.items()):
        # print(image.shape,' ',type(image))
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace("_", " ").title(), fontsize=20)
        plt.imshow(image)
    plt.show()
