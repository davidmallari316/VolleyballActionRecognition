import os

import torch
from torchvision.io import write_png
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.utils import draw_bounding_boxes
from torchcodec.decoders import VideoDecoder
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


class ActionRecognizer:
    """Wraps a pretrained video model for action recognition on clips."""

    def __init__(self, device=None, top_k=5):
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

    @torch.no_grad()
    def _person_box(self, frame, min_score=0.5):
        """Union box (x1, y1, x2, y2) of all people in a (C, H, W) uint8 frame, or None."""
        det = self.detector([frame.float().div(255).to(self.device)])[0]
        keep = (det["labels"] == self.person_id) & (det["scores"] >= min_score)
        boxes = det["boxes"][keep]
        if len(boxes) == 0:
            return None
        return (
            boxes[:, 0].min().item(), boxes[:, 1].min().item(),
            boxes[:, 2].max().item(), boxes[:, 3].max().item(),
        )

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