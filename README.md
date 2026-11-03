# VolleyballActionRecognition

## Fine-tuning the detector on a Roboflow dataset

1. Download the dataset as **COCO JSON** with your Roboflow API key (`export ROBOFLOW_API_KEY=...`):

   ```python
   from LoadDataset import LoadDataset
   paths = LoadDataset("my-workspace", "my-project", 1).download()  # -> datasets/my-project-v1/
   ```
   or `python LoadDataset.py my-workspace my-project 1`.
2. Train (keeps the best checkpoint by validation COCO mAP):

   ```python
   from ActionRecognizer import ActionRecognizer
   r = ActionRecognizer()
   r.train_detector(paths["train"], paths["valid"], epochs=10)
   ```
3. Reload for inference: `ActionRecognizer(detector_checkpoint="checkpoints/detector.pt")`, then use `detect(frame)` for action boxes.
