import glob
import json
import logging
import os

import cv2
import numpy as np
import torch
import tqdm

# import utils.common as common
project_root = os.path.abspath(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    )
)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, name="SOC", is_train=False, transform=None):
        self.is_train = is_train
        self.name = name
        self.images = []
        self.labels = []
        self.serial_number = []
        self.transform = transform
        self._load_data(path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        _image = self.images[idx]
        _label = self.labels[idx]
        _serial_number = self.serial_number[idx]

        if self.transform:
            _image = self.transform(_image)

        return _image, _label, _serial_number

    def _load_data(self, path):
        mode = "train" if self.is_train else "test"
        folder_pattern = "*" if self.name == "all" else self.name

        search_path_img = os.path.join(path, folder_pattern, mode, "**/*.png")
        search_path_json = os.path.join(path, folder_pattern, mode, "**/*.json")

        image_list = glob.glob(search_path_img, recursive=True)
        label_list = glob.glob(search_path_json, recursive=True)

        # Important : trier pour garantir la correspondance image/label
        image_list.sort()
        label_list.sort()

        for image_path, label_path in tqdm.tqdm(
            zip(image_list, label_list),
            desc=f"Loading {self.name} ({mode})",
            total=len(label_list),
        ):
            if (
                os.path.splitext(os.path.basename(image_path))[0]
                != os.path.splitext(os.path.basename(label_path))[0]
            ):
                continue

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # dim 2 if it's png
            if image is None:
                logging.error(f"Failed to load image: {image_path}")
                continue

            if len(image.shape) < 3:
                image = np.expand_dims(image, axis=0)  # add channel dim

            image_tensor = torch.tensor(image, dtype=torch.float32)
            self.images.append(image_tensor)
            # self.images.append(np.load(image_path))

            with open(label_path, mode="r", encoding="utf-8") as f:
                _label = json.load(f)

            self.labels.append(_label["class_id"])
            self.serial_number.append(_label["serial_number"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Configuration
    DATA_PATH = "datasets/MSTAR_IMG_JSON"  # Adjust if your path is different
    BATCH_SIZE = 32

    logging.info("-" * 30)
    logging.info("Testing Dataset: Single Split (SOC)")
    logging.info("-" * 30)

    try:
        # 1. Test Single Split (Standard)
        dataset_soc = Dataset(path=DATA_PATH, name="SOC", is_train=True)
        logging.info(f"SOC Train size: {len(dataset_soc)}")

        # Wrap in DataLoader to check batching compatibility
        loader = torch.utils.data.DataLoader(
            dataset_soc, batch_size=BATCH_SIZE, shuffle=True
        )
        img, label, sn = next(iter(loader))
        logging.info(f"Batch shapes -> Images: {img.shape}, Labels: {label.shape}")
        logging.info(f"Sample Serial Number: {sn[0]}")

    except Exception as e:
        logging.error(f"Error loading SOC: {e}")

    logging.info("\n" + "-" * 30)
    logging.info("Testing Dataset: Combined Split (all)")
    logging.info("-" * 30)

    try:
        # 2. Test Combined Split
        dataset_all = Dataset(path=DATA_PATH, name="all", is_train=True)
        logging.info(f"Combined 'all' Train size: {len(dataset_all)}")

        if len(dataset_all) > 0:
            # Quick check on class distribution
            from collections import Counter

            dist = Counter(dataset_all.labels)
            logging.info(f"Class distribution (Top 5): {dist.most_common(5)}")

            # Final integrity check
            if len(dataset_all) > len(dataset_soc):
                logging.info("Success: 'all' contains more samples than 'SOC' alone.")
            else:
                logging.warning(
                    "Warning: 'all' size is not greater than 'SOC'. Check paths."
                )
        else:
            logging.error("❌ Error: No data found for 'all'.")

    except Exception as e:
        logging.error(f"Error loading 'all': {e}")
