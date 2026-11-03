import json
import os

import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import v2
from torchvision import tv_tensors


class RoboflowDetectionDataset(Dataset):
    """
    Roboflow COCO-JSON export split (e.g. datasets/volleyball/train).

    Each item is (image uint8 CHW, target) where target has xyxy `boxes`,
    int64 `labels` (>= 1, 0 is reserved for background) and `image_id`.
    """

    def __init__(self, root, train=False):
        self.root = root
        with open(os.path.join(root, "_annotations.coco.json")) as f:
            coco = json.load(f)

        # Roboflow exports usually include a dummy category with id 0 (the
        # dataset name). Drop it and remap real classes to contiguous 1..N.
        cats = sorted(coco["categories"], key=lambda c: c["id"])
        real = cats[1:] if len(cats) > 1 and cats[0]["id"] == 0 else cats
        self.class_names = [c["name"] for c in real]
        self._cat_to_label = {c["id"]: i + 1 for i, c in enumerate(real)}

        self.images = {im["id"]: im for im in coco["images"]}
        self.ids = sorted(self.images)
        self.anns = {i: [] for i in self.ids}
        for a in coco["annotations"]:
            if a["category_id"] in self._cat_to_label:
                self.anns[a["image_id"]].append(a)

        self.transform = (
            v2.Compose([
                v2.RandomHorizontalFlip(0.5),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                v2.SanitizeBoundingBoxes(),
            ])
            if train else None
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        info = self.images[image_id]
        image = read_image(os.path.join(self.root, info["file_name"]), ImageReadMode.RGB)

        boxes, labels = [], []
        for a in self.anns[image_id]:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self._cat_to_label[a["category_id"]])

        boxes_t = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        target = {
            "boxes": tv_tensors.BoundingBoxes(
                boxes_t, format="XYXY", canvas_size=tuple(image.shape[-2:])
            ),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": image_id,
        }
        if self.transform is not None:
            image, target = self.transform(image, target)
        target["boxes"] = target["boxes"].as_subclass(torch.Tensor)
        return image.as_subclass(torch.Tensor), target


def collate_fn(batch):
    return tuple(zip(*batch))
