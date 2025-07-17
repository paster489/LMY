from comet_ml import Experiment

import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import random
import numpy as np
import torch

import cv2


import rasterio
from torch.utils.data import Dataset
import albumentations as A
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

from sklearn.metrics import jaccard_score
from torch.nn.functional import interpolate

# REPRODUCIBILITY FUNCTION
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # if you ever switch to GPU
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(42)

# COMET
experiment = Experiment(
    api_key=os.environ.get("COMET_API_KEY"),
    project_name="building-segmentation"
)

# Dataset class
class SegFormerTIFFDataset(Dataset):
    def __init__(self, img_dir, mask_dir, processor, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.transform = transform
        self.ids = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, img_id)
        mask_path = os.path.join(self.mask_dir, img_id.replace(".tif", ".png"))

        with rasterio.open(img_path) as src:
            image = src.read([1, 2, 3])

            image = np.transpose(image, (1, 2, 0)).astype(np.float32)
            image = 255 * (image - image.min()) / (image.max() - image.min() + 1e-8)
            image = image.astype(np.uint8)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # Handle missing or corrupted mask by creating an all-background mask
            # print(f"[WARN] Mask not found or unreadable: {mask_path}. Replacing with empty mask.")
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        inputs = self.processor(images=image, segmentation_maps=mask, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].squeeze()
        inputs["labels"] = inputs["labels"].squeeze()
        return inputs

# Albumentations with reproducibility
def get_train_augmentations():
    return A.Compose([
            A.HorizontalFlip(p=0.5),             # Common, helps with symmetry invariance
            A.RandomRotate90(p=0.5),             # Common for square patches (90, 180, 270°)
            A.RandomBrightnessContrast(p=0.3),   # Helps model generalize lighting/shadow
            A.Resize(512, 512),                  # Required to match model input size
    ], additional_targets={"mask": "mask"})

def get_val_augmentations():
    return A.Compose([
        A.Resize(512, 512),
    ], additional_targets={"mask": "mask"})




def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Convert logits to predictions
    preds = np.argmax(logits, axis=1)

    # Resize predictions if shape mismatch
    if preds.shape != labels.shape:
        # Convert to torch tensors to use interpolate
        preds_t = torch.tensor(preds).unsqueeze(1).float()  # [B,1,H,W]
        resized_preds = interpolate(preds_t, size=labels.shape[-2:], mode='nearest')  # Match label size
        preds = resized_preds.squeeze().long().numpy()

    # Flatten
    preds = preds.flatten()
    labels = labels.flatten()

    # Compute binary IoU
    iou = jaccard_score(labels, preds, average='binary')

    return {"iou": iou}

# Main
if __name__ == "__main__":
    processor = SegformerImageProcessor(
        reduce_labels=False, do_resize=True, size=(512, 512), do_normalize=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=2,
        ignore_mismatched_sizes=True
    ).to(device)

    train_dataset = SegFormerTIFFDataset(
        img_dir="/home/paster/LMY/data/general/split/train/images",
        mask_dir="/home/paster/LMY/data/general/split/train/mask",
        processor=processor,
        transform=get_train_augmentations()
    )

    val_dataset = SegFormerTIFFDataset(
        img_dir="/home/paster/LMY/data/general/split/val/images",
        mask_dir="/home/paster/LMY/data/general/split/val/mask",
        processor=processor,
        transform=get_val_augmentations()
    )


    training_args = TrainingArguments(
        output_dir="/home/paster/LMY/segformer_output", #Where to save checkpoints, logs, and the final model.
        learning_rate=5e-5, # Small values like this are best for fine-tuning transformers.
        per_device_train_batch_size=2, #  Small batch sizes are often used when memory is limited
        num_train_epochs=30,
        eval_strategy="epoch", # Run evaluation after every epoch
        logging_steps=10, # Log metrics (like loss) every 10 steps. Good for tracking training progress
        save_strategy="epoch", # Save model checkpoints after every epoch.
        save_total_limit=2, # Keep only the 2 most recent checkpoints to save disk space. Older ones are deleted.
        load_best_model_at_end=True, # After training, automatically reload the best model (based on validation score)
        metric_for_best_model="iou", # Use "iou" (Intersection over Union) as the metric to decide which checkpoint is best.
        greater_is_better=True, # Set to True because higher IoU means better performance. If you're tracking loss, you'd set it to False.
        remove_unused_columns=False, # Prevents removal of unused columns in the dataset — important for custom dataset formats
        report_to="comet_ml", # Send logs and metrics to Comet.ml (your experiment tracker).
        run_name="segformer-iou-selection", # The name of this training run — shown in logs and Comet.
        no_cuda=False, # if True - forces training on CPU only. Set to False to use GPU (recommended if available).
        seed=42, # Sets random seed for reproducibility.
        data_seed=42, # Specifically seeds data shuffling, like train/val splits.
        disable_tqdm=False # Show live progress bars (TQDM). Set to True to disable in non-interactive environments.
    )

    experiment.set_name("SegFormer")
    experiment.log_parameters({
        "model": "SegFormer-B0",
        "epochs": 100,
        "batch_size": 2,
        "learning_rate": 5e-5,
        "train_aug": "flip, rotate90, contrast, resize",
        "val_aug": "resize only",
        "device": "GPU",
        "seed": 42
    })

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train()