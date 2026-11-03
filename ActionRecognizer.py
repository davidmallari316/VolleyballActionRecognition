import os

import torch
from torchvision.io import write_png
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes
from torchcodec.decoders import VideoDecoder
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

from RoboflowDataset import RoboflowDetectionDataset, collate_fn


class ActionRecognizer:
    """Wraps a pretrained video model for action recognition on clips."""

    def __init__(self, device=None, top_k=5, class_names=None, detector_checkpoint=None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.top_k = top_k

        self.weights = R2Plus1D_18_Weights.KINETICS400_V1
        self.model = r2plus1d_18(weights=self.weights)
        self.model.eval().to(self.device)

        # Preprocessing bundled with the weights: resize, crop, normalize.
        self.transform = self.weights.transforms()
        self.categories = self.weights.meta["categories"]

        # Person detector used to localize the action (the classifier has no boxes).
        det_weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.detector = fasterrcnn_resnet50_fpn(weights=det_weights)
        self.detector.eval().to(self.device)
        self.person_id = det_weights.meta["categories"].index("person")

        # Fine-tuned detector: swap the COCO box head for volleyball action classes.
        # detector_classes is None while the stock COCO person detector is in use.
        self.detector_classes = None
        if detector_checkpoint is not None:
            ckpt = torch.load(detector_checkpoint, map_location=self.device)
            self._set_detector_classes(ckpt["class_names"])
            self.detector.load_state_dict(ckpt["state_dict"])
        elif class_names is not None:
            self._set_detector_classes(class_names)
        self.detector.eval().to(self.device)

    def _set_detector_classes(self, class_names):
        """Replace the detector's box head with one for `class_names` (+ background)."""
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, len(class_names) + 1
        )
        self.detector_classes = ["__background__", *class_names]
        self.detector.to(self.device)

    @torch.no_grad()
    def detect(self, frame, min_score=0.5):
        """
        Fine-tuned action detections for a (C, H, W) uint8 frame.

        Returns a list of (label, score, (x1, y1, x2, y2)), highest score first.
        """
        if self.detector_classes is None:
            raise RuntimeError("No fine-tuned detector: pass class_names or detector_checkpoint.")
        det = self.detector([frame.float().div(255).to(self.device)])[0]
        return [
            (self.detector_classes[l], s.item(), tuple(b.tolist()))
            for b, l, s in zip(det["boxes"], det["labels"], det["scores"])
            if s >= min_score
        ]

    @torch.no_grad()
    def _person_box(self, frame, min_score=0.5):
        """
        Union box (x1, y1, x2, y2) of all people in a (C, H, W) uint8 frame, or None.
        With a fine-tuned detector, it is the union of all detected action boxes.
        """
        det = self.detector([frame.float().div(255).to(self.device)])[0]
        keep = det["scores"] >= min_score
        if self.detector_classes is None:
            keep &= det["labels"] == self.person_id
        boxes = det["boxes"][keep]
        if len(boxes) == 0:
            return None
        return (
            boxes[:, 0].min().item(), boxes[:, 1].min().item(),
            boxes[:, 2].max().item(), boxes[:, 3].max().item(),
        )

    def train_detector(
        self,
        train_dir,
        val_dir,
        epochs=10,
        batch_size=4,
        lr=0.005,
        freeze_backbone=True,
        save_path="checkpoints/detector.pt",
    ):
        """
        Fine-tune the Faster R-CNN on a Roboflow COCO-JSON export.

        Keeps the checkpoint with the best validation mAP (COCO, IoU 0.5:0.95)
        at `save_path`, saved with its class names. Returns that best mAP.
        """
        from torchmetrics.detection import MeanAveragePrecision

        train_ds = RoboflowDetectionDataset(train_dir, train=True)
        val_ds = RoboflowDetectionDataset(val_dir, train=False)
        if self.detector_classes is None or self.detector_classes[1:] != train_ds.class_names:
            self._set_detector_classes(train_ds.class_names)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_fn)

        self.detector.backbone.body.requires_grad_(not freeze_backbone)
        params = [p for p in self.detector.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        best_map = -1.0
        for epoch in range(epochs):
            self.detector.train()
            total = 0.0
            for images, targets in train_loader:
                images = [i.float().div(255).to(self.device) for i in images]
                targets = [
                    {"boxes": t["boxes"].to(self.device), "labels": t["labels"].to(self.device)}
                    for t in targets
                ]
                loss = sum(self.detector(images, targets).values())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
            scheduler.step()

            # Validation: eval mode returns predictions, scored with COCO mAP.
            self.detector.eval()
            metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
            with torch.no_grad():
                for images, targets in val_loader:
                    preds = self.detector([i.float().div(255).to(self.device) for i in images])
                    metric.update(
                        [{k: v.cpu() for k, v in p.items()} for p in preds],
                        [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets],
                    )
            scores = metric.compute()
            val_map = scores["map"].item()
            print(
                f"epoch {epoch + 1}/{epochs} loss={total / max(len(train_loader), 1):.4f} "
                f"mAP={val_map:.4f} mAP50={scores['map_50'].item():.4f}"
            )
            if val_map > best_map:
                best_map = val_map
                torch.save(
                    {"state_dict": self.detector.state_dict(),
                     "class_names": self.detector_classes[1:]},
                    save_path,
                )

        # Leave the detector holding the best weights, in eval mode.
        self.detector.load_state_dict(torch.load(save_path, map_location=self.device)["state_dict"])
        self.detector.eval()
        return best_map

    @torch.no_grad()
    def predict(self, frames):
        """
        Classify a single video clip.

        Args:
            frames: uint8 tensor of shape (T, C, H, W), RGB, as TorchCodec returns.

        Returns:
            List of (label, probability, box) tuples, highest first. box is
            (x1, y1, x2, y2) in original frame pixels around the people in the
            clip's middle frame, or None if no one was detected.
        """
        # The transform takes (T, C, H, W) and returns (C, T, H, W).
        clip = self.transform(frames).unsqueeze(0).to(self.device)  

        logits = self.model(clip)
        probs = logits.softmax(dim=1)[0]

        values, indices = probs.topk(self.top_k)
        box = self._person_box(frames[len(frames) // 2])
        return [(self.categories[i], v.item(), box) for v, i in zip(values, indices)]

    def predict_video(self, source, clip_len=16, stride=8):
        """
        Slide a window over a video and return predictions per window.

        Only the frames for each window are decoded, so long videos don't
        need to fit in memory.

        Args:
            source: path to a video file, or an existing VideoDecoder.
            clip_len: number of frames per window.
            stride: step between window starts.

        Returns:
            List of (start_frame, top_predictions) tuples.
        """
        decoder = source if isinstance(source, VideoDecoder) else VideoDecoder(source)
        total = decoder.metadata.num_frames

        results = []
        for start in range(0, max(total - clip_len + 1, 1), stride):
            stop = min(start + clip_len, total)
            frames = decoder.get_frames_in_range(start, stop).data  # (T, C, H, W)
            results.append((start, self.predict(frames)))
        return results


if __name__ == "__main__":
    # Fine-tune the detector on a Roboflow COCO export, then reload it:
    # recognizer = ActionRecognizer()
    # recognizer.train_detector("datasets/volleyball/train", "datasets/volleyball/valid", epochs=10)
    # recognizer = ActionRecognizer(detector_checkpoint="checkpoints/detector.pt")

    decoder = VideoDecoder("assets/rally_women.mp4")
    recognizer = ActionRecognizer(top_k=3)

    # Classify the first 16 frames
    first_clip = decoder.get_frames_in_range(0, 16).data
    os.makedirs("assets/output", exist_ok=True)
    for i, (label, prob, box) in enumerate(recognizer.predict(first_clip)):
        text = f"{label}:{prob:.3f}"
        print(text)
        # Save the clip frame matching this prediction's rank, named after the printed line.
        image = first_clip[i]
        if box is not None:
            image = draw_bounding_boxes(
                image, torch.tensor([box]), labels=[text], colors="red", width=3
            )
        write_png(image, f"assets/output/{text}.png")

    # # Or scan the whole video
    # for start, preds in recognizer.predict_video(decoder, clip_len=16, stride=16):
    #     top_label, top_prob, _ = preds[0]
    #     print(f"frames {start}-{start + 16}: {top_label} ({top_prob:.2f})")