import torch
import torch.nn as nn
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

    @torch.no_grad()
    def predict(self, clip):
        """
        Classify a single video clip.

        Args:
            clip: uint8 tensor of shape (T, H, W, C) or (C, T, H, W),
                  with T frames. Frames should be RGB.

        Returns:
            List of (label, probability) tuples, highest first.
        """
        clip = self._to_tchw(clip)
        clip = self.transform(clip).unsqueeze(0).to(self.device)  # (1, C, T, H, W)

        logits = self.model(clip)
        probs = logits.softmax(dim=1)[0]

        values, indices = probs.topk(self.top_k)
        return [(self.categories[i], v.item()) for v, i in zip(values, indices)]

    def predict_video(self, frames, clip_len=16, stride=8):
        """
        Slide a window over a long video and return predictions per window.

        Args:
            frames: uint8 tensor of shape (T, H, W, C) for the whole video.
            clip_len: number of frames per window.
            stride: step between window starts.

        Returns:
            List of (start_frame, top_predictions) tuples.
        """
        results = []
        total = frames.shape[0]
        for start in range(0, max(total - clip_len + 1, 1), stride):
            window = frames[start:start + clip_len]
            results.append((start, self.predict(window)))
        return results

    @staticmethod
    def _to_tchw(clip):
        # The torchvision Kinetics preset expects (T, C, H, W) frames.
        if clip.ndim != 4:
            raise ValueError(f"Expected 4D clip tensor, got shape {tuple(clip.shape)}")
        if clip.shape[-1] == 3:  # (T, H, W, C) -> (T, C, H, W)
            clip = clip.permute(0, 3, 1, 2)
        elif clip.shape[0] == 3:  # (C, T, H, W) -> (T, C, H, W)
            clip = clip.permute(1, 0, 2, 3)
        return clip


if __name__ == "__main__":
    import torchvision.io as io

    # Load a video: returns (T, H, W, C) uint8 frames.
    video, _, _ = io.read_video("example.mp4", pts_unit="sec")

    recognizer = ActionRecognizer(top_k=3)

    # Classify the first 16 frames
    for label, prob in recognizer.predict(video[:16]):
        print(f"{label}: {prob:.3f}")

    # Or scan the whole video
    for start, preds in recognizer.predict_video(video, clip_len=16, stride=16):
        top_label, top_prob = preds[0]
        print(f"frames {start}-{start + 16}: {top_label} ({top_prob:.2f})")