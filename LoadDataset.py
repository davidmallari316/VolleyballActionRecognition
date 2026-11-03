import argparse
import os

ANNOTATIONS = "_annotations.coco.json"


class LoadDataset:
    """Downloads a Roboflow dataset as COCO JSON, in the layout RoboflowDetectionDataset reads."""

    def __init__(self, workspace, project, version, api_key=None, root="datasets", fmt="coco"):
        self.workspace = workspace
        self.project = project
        self.version = version
        # Never stored on disk or printed.
        self._api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
        self.root = root
        self.fmt = fmt
        self.location = os.path.join(root, f"{project}-v{version}")

    def _split_dir(self, *names):
        """First existing split dir (with annotations) among `names`, or None."""
        for name in names:
            path = os.path.join(self.location, name)
            if os.path.isfile(os.path.join(path, ANNOTATIONS)):
                return path
        return None

    def _paths(self):
        train = self._split_dir("train")
        valid = self._split_dir("valid", "val")
        if train is None or valid is None:
            missing = [n for n, p in (("train", train), ("valid", valid)) if p is None]
            raise FileNotFoundError(
                f"Export at {self.location} is missing {missing} (expected "
                f"<split>/{ANNOTATIONS}). Is the dataset exported as {self.fmt!r}?"
            )
        return {
            "root": self.location,
            "train": train,
            "valid": valid,
            "test": self._split_dir("test"),
        }

    def download(self, overwrite=False):
        """
        Download the dataset (skipped if already present) and return split paths:
        {"root", "train", "valid", "test" (None if absent)}.
        """
        if not overwrite and self._split_dir("train") is not None:
            print(f"Dataset already at {self.location}, skipping download.")
            return self._paths()

        if not self._api_key:
            raise ValueError(
                "No Roboflow API key: pass api_key or set the ROBOFLOW_API_KEY env var."
            )

        from roboflow import Roboflow  # lazy: only needed when downloading

        rf = Roboflow(api_key=self._api_key)
        version = rf.workspace(self.workspace).project(self.project).version(self.version)
        version.download(self.fmt, location=self.location, overwrite=overwrite)
        return self._paths()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a Roboflow dataset as COCO JSON.")
    parser.add_argument("workspace")
    parser.add_argument("project")
    parser.add_argument("version", type=int)
    parser.add_argument("--root", default="datasets")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = LoadDataset(args.workspace, args.project, args.version, root=args.root).download(
        overwrite=args.overwrite
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
