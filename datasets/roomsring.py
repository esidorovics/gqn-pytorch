import os, gzip
import torch
from torch.utils.data import Dataset


def transform_viewpoint(v):
    """
    Transforms the viewpoint vector into a consistent
    representation
    """
    w, z = torch.split(v, 3, dim=-1)
    y, p = torch.split(z, 1, dim=-1)

    # position, [yaw, pitch]
    view_vector = [w, torch.cos(y), torch.sin(y), torch.cos(p), torch.sin(p)]
    v_hat = torch.cat(view_vector, dim=-1)

    return v_hat


class RoomsRingCamera(Dataset):
    """
    RoomsRing dataset. Based on the dataset provided
    in the GQN paper.
    :param root_dir: location of data on disc
    :param train: whether to use train of test set
    :param transform: transform on images
    :param target_transform: transform on viewpoints
    """
    def __init__(self, root_dir, train=True, transform=None, target_transform=transform_viewpoint):
        super(RoomsRingCamera, self).__init__()
        prefix = "train" if train else "test"
        self.root_dir = os.path.join(root_dir, prefix)
        self.records = sorted([p for p in os.listdir(self.root_dir) if "pt" in p])
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        scene_path = os.path.join(self.root_dir, self.records[idx])

        with gzip.open(scene_path, "r") as f:
            data = torch.load(f)

        images, viewpoints = list(zip(*data))

        # (b, m, c, h, w)
        images = torch.FloatTensor(images)
        if self.transform:
            images = self.transform(images)

        # (b, m, 5)
        viewpoints = torch.FloatTensor(viewpoints)
        if self.target_transform:
            viewpoints = self.target_transform(viewpoints)

        return images, viewpoints

if __name__ == '__main__':
    import cv2
    import torchvision
    import numpy as np
    from PIL import Image
    train_dataset = RoomsRingCamera(root_dir="../data/rooms_ring_camera")
    valid_dataset = RoomsRingCamera(root_dir="../data/rooms_ring_camera", train=False)
    img, v = train_dataset[5]
    import torchvision
    for i in range(10):
        _img = img[34, i, :, :, :]
        # _img = torchvision.transforms.ToPILImage()(_img)
        # _img.save("../data/test_images/roomsring{}.png".format(i))
        cv2.imwrite("../data/test_images/roomsring{}.png".format(i), _img.permute(1, 2, 0).numpy())

