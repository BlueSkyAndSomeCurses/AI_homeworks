"""
EfficientNet-B2 + UNet++ Training Script
Trains EfficientNet-B2 encoder with UNet++ decoder for MRI bowel/stomach segmentation.
Uses 2.5D approach with combined Dice + BCE loss.
"""

import numpy as np
import random
from pathlib import Path
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import polars as pl
from PIL import Image
import albumentations as A
import cv2
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
import segmentation_models_pytorch as smp

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# ============================================================================
# DATA PREPROCESSING FUNCTIONS
# ============================================================================


def rle_to_mask(rle: str, height: int, width: int) -> np.ndarray:
    """Convert RLE string to binary mask."""
    s = list(map(int, rle.split()))
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((height, width))


def mask_to_rle(mask: np.ndarray) -> str:
    """Convert binary mask to RLE string."""
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def get_image_and_masks(
    data: pl.DataFrame, case: int, day: int, slice: int, images_path: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load image and masks for a specific case, day, and slice."""
    image_masks = (
        data.filter(
            pl.col("case").eq(case) & pl.col("day").eq(day) & pl.col("slice").eq(slice)
        )
        .group_by("case", "day", "slice")
        .agg(pl.col("class"), pl.col("segmentation"))
        .explode("class", "segmentation")
    )

    image = np.array(
        Image.open(f"{images_path}/case{case}/day{day}/slice_{slice}.png"),
        dtype=np.float32,
    )

    large_bowel_rle = image_masks.filter(pl.col("class").eq("large_bowel"))[
        "segmentation"
    ][0]

    small_bowel_rle = image_masks.filter(pl.col("class").eq("small_bowel"))[
        "segmentation"
    ][0]

    stomach_rle = image_masks.filter(pl.col("class").eq("stomach"))["segmentation"][0]

    return (
        image,
        rle_to_mask(large_bowel_rle, *image.shape),
        rle_to_mask(small_bowel_rle, *image.shape),
        rle_to_mask(stomach_rle, *image.shape),
    )


def preprocess_mri(image_uint16: np.ndarray) -> np.ndarray:
    """Normalize MRI image to [0, 1] range."""
    image = image_uint16.astype(np.float32)
    image /= image.max()
    return image


# ============================================================================
# AUGMENTATION FUNCTIONS
# ============================================================================


def get_train_augmentations(img_size=(288, 288)) -> A.Compose:
    """Get training augmentation pipeline."""
    train_tfms = A.Compose(
        [
            A.LongestMaxSize(max_size=max(img_size)),
            A.PadIfNeeded(*img_size, border_mode=cv2.BORDER_CONSTANT, fill=0),
            A.RandomScale(scale_limit=(-0.2, 0.5), p=1, interpolation=1),
            A.PadIfNeeded(*img_size, border_mode=cv2.BORDER_CONSTANT, fill=0),
            A.RandomCrop(*img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=1),
        ]
    )
    return train_tfms


def get_test_augmentations(img_size=(288, 288)) -> A.Compose:
    """Get test/validation augmentation pipeline."""
    valid_tfms = A.Compose(
        [
            A.LongestMaxSize(max_size=max(img_size)),
            A.PadIfNeeded(*img_size, border_mode=cv2.BORDER_CONSTANT, fill=0),
        ]
    )
    return valid_tfms


# ============================================================================
# DATASET CLASS
# ============================================================================


class MRISegmentationDataset(Dataset):
    """Dataset class for MRI segmentation with 2.5D support."""

    def __init__(
        self,
        dataframe,
        images_path: str,
        preprocess_fn=None,
        augmentations=None,
        two_point_five_d: bool = False,
        depth: int = 3,
    ):
        self.dataframe = dataframe
        self.unique_samples = dataframe.select("case", "day", "slice").unique()
        self.preprocess_fn = preprocess_fn
        self.augmentations = augmentations
        self.images_path = images_path
        self.two_point_five_d = two_point_five_d
        self.depth = depth if depth % 2 == 1 else depth + 1

    def __len__(self):
        return len(self.unique_samples)

    def __getitem__(self, idx):
        row = self.unique_samples[idx]
        case, day, slice_idx = (
            int(row["case"].item()),
            int(row["day"].item()),
            int(row["slice"].item()),
        )

        image, large_mask, small_mask, stomach_mask = get_image_and_masks(
            self.dataframe, case, day, slice_idx, self.images_path
        )

        if self.two_point_five_d:
            image = self._get_2p5d_stack(case, day, slice_idx)

        mask = np.stack([large_mask, small_mask, stomach_mask], axis=-1).astype(
            np.float32
        )

        if self.preprocess_fn:
            if self.two_point_five_d:
                image = np.stack([self.preprocess_fn(im) for im in image], axis=0)
            else:
                image = self.preprocess_fn(image)

        if self.augmentations:
            if self.two_point_five_d:
                image_hw_c = np.transpose(image, (1, 2, 0))
                augmented = self.augmentations(image=image_hw_c, mask=mask)
                image = np.transpose(augmented["image"], (2, 0, 1))
                mask = augmented["mask"]
            else:
                augmented = self.augmentations(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]

        mask = np.transpose(mask, (2, 0, 1))

        if image.ndim == 2:
            image = np.expand_dims(image, 0)

        image_tensor = torch.tensor(image, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)

        return image_tensor, mask_tensor, case, day, slice_idx

    def _get_2p5d_stack(self, case, day, center_slice_idx):
        """Get 2.5D stack of slices around center slice."""
        half = self.depth // 2

        slices_df = (
            self.dataframe.filter(
                (self.dataframe["case"] == case) & (self.dataframe["day"] == day)
            )
            .select("slice")
            .unique()
            .sort("slice")
        )
        all_slices = slices_df["slice"].to_list()
        num_slices = len(all_slices)

        center_pos = all_slices.index(center_slice_idx)

        idxs = [
            all_slices[np.clip(center_pos + o, 0, num_slices - 1)]
            for o in range(-half, half + 1)
        ]

        images = []
        for s in idxs:
            im, _, _, _ = get_image_and_masks(
                self.dataframe, case, day, s, self.images_path
            )
            images.append(im)

        return np.stack(images, axis=0).astype(np.float32)


# ============================================================================
# LOSS FUNCTION
# ============================================================================


class CombinedLoss(nn.Module):
    """Combined Dice + BCE loss for multi-label segmentation."""

    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = smp.losses.DiceLoss(mode="multilabel")
        self.bce_loss = smp.losses.SoftBCEWithLogitsLoss()

    def forward(self, outputs, targets):
        dice = self.dice_loss(outputs, targets)
        bce = self.bce_loss(outputs, targets)
        return self.dice_weight * dice + self.bce_weight * bce


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================


def dice_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    """Calculate Dice coefficient."""
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=(1, 0))
    return dice


def iou_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    """Calculate IoU coefficient."""
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)
    iou = ((inter + epsilon) / (union + epsilon)).mean(dim=(1, 0))
    return iou


def train_one_epoch(model, train_loader, loss_fn, optimizer, device, scheduler=None):
    """Train model for one epoch."""
    model.train()
    epoch_loss = 0.0

    print("Training one epoch")
    pbar = tqdm(train_loader, desc="Train ")
    for batch in pbar:
        images, masks, *_ = batch

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        epoch_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    epoch_loss /= len(train_loader)
    return epoch_loss


def evaluate(model, val_loader, device, thr: float = 0.5):
    """Evaluate model on validation set."""
    model.eval()
    dices = []
    ious = []

    print("Evaluating model")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            images, masks, *_ = batch
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            outputs_soft = torch.sigmoid(outputs)

            dice = dice_coef(masks, outputs_soft, thr).cpu()
            iou = iou_coef(masks, outputs_soft, thr).cpu()
            ious.append(iou)
            dices.append(dice)

    return np.array(dices).mean(), np.array(ious).mean()


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience=5, mode="max", delta=1e-4, save_path=None):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.save_path = Path(save_path) if save_path else None

    def __call__(self, current_score, model):
        if self.best_score is None:
            self.best_score = current_score
            if self.save_path:
                torch.save(model.state_dict(), self.save_path)
            return False

        improvement = (
            current_score > self.best_score + self.delta
            if self.mode == "max"
            else current_score < self.best_score - self.delta
        )

        if improvement:
            self.best_score = current_score
            self.counter = 0
            if self.save_path:
                torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def get_data_loaders(
    full_dataset: pl.DataFrame,
    fold: int,
    train_images_path: str,
    two_point_five_d: bool = False,
    depth: int = 3,
    batch_size: int = 8,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders for a specific fold."""
    train_dataframe = full_dataset.filter(pl.col("fold").ne(fold))
    val_dataframe = full_dataset.filter(pl.col("fold").eq(fold))

    train_data = MRISegmentationDataset(
        train_dataframe,
        images_path=train_images_path,
        preprocess_fn=preprocess_mri,
        augmentations=get_train_augmentations(),
        two_point_five_d=two_point_five_d,
        depth=depth,
    )
    val_data = MRISegmentationDataset(
        val_dataframe,
        images_path=train_images_path,
        preprocess_fn=preprocess_mri,
        augmentations=get_test_augmentations(),
        two_point_five_d=two_point_five_d,
        depth=depth,
    )

    return DataLoader(train_data, batch_size=batch_size, shuffle=True), DataLoader(
        val_data, batch_size=batch_size, shuffle=False
    )


def run_training(
    full_data: pl.DataFrame,
    model_builder,
    device,
    train_images_path: str,
    folds_num: int = 4,
    epochs: int = 50,
    save_model_path: str = "models",
    model_name: str = "undefined",
    patience: int = 5,
    two_point_five_d: bool = False,
    depth: int = 3,
    batch_size: int = 8,
    scheduler_type: str = "step",
):
    """Run complete training loop for all folds."""
    models, loss_fns, optimizers = model_builder(device, folds_num)

    save_dir = Path(save_model_path) / model_name
    save_dir.mkdir(exist_ok=True, parents=True)

    for fold in range(folds_num):
        print(f"\n{'=' * 60}")
        print(f"Training fold {fold}")
        print(f"{'=' * 60}\n")

        model = models[fold]
        loss_fn = loss_fns[fold]
        optimizer = optimizers[fold]

        train_loader, val_loader = get_data_loaders(
            full_data,
            fold,
            train_images_path,
            two_point_five_d=two_point_five_d,
            depth=depth,
            batch_size=batch_size,
        )

        early_stopping = EarlyStopping(
            patience=patience, mode="max", save_path=save_dir / f"fold_{fold}.pth"
        )

        scheduler = None
        if scheduler_type == "onecycle":
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=optimizer.param_groups[0]["lr"] * 10,
                epochs=epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.3,
                anneal_strategy="cos",
            )
        elif scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=1e-6,
            )
        elif scheduler_type == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=max(1, epochs // 3),
                gamma=0.1,
            )

        print(f"Starting training with {scheduler_type} scheduler")
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                device,
                scheduler if scheduler_type == "onecycle" else None,
            )
            val_dice, val_iou = evaluate(model, val_loader, device)

            if scheduler is not None and scheduler_type in ["cosine", "step"]:
                scheduler.step()
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | "
                    f"Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f} | LR: {current_lr:.6f}"
                )
            else:
                print(
                    f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | "
                    f"Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}"
                )

            if early_stopping(val_dice, model):
                print(f"Early stopping at epoch {epoch}")
                break

    return models


# ============================================================================
# MODEL BUILDER
# ============================================================================


def build_efficientnet_unetpp(device, folds_num=4):
    """Build EfficientNet-B2 + UNet++ model with combined loss."""
    ENCODER = "efficientnet-b2"
    ENCODER_WEIGHTS = "imagenet"
    IN_CHANNELS = 5  # For 2.5D with depth=5
    NUM_CLASSES = 3  # large_bowel, small_bowel, stomach
    LR = 1e-3
    WEIGHT_DECAY = 1e-6

    models = [
        smp.UnetPlusPlus(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=IN_CHANNELS,
            classes=NUM_CLASSES,
            activation=None,
        ).to(device)
        for _ in range(folds_num)
    ]

    loss_fns = [
        CombinedLoss(dice_weight=0.5, bce_weight=0.5).to(device)
        for _ in range(folds_num)
    ]

    optimizers = [
        optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        for model in models
    ]

    return models, loss_fns, optimizers


# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================


def dir_to_dataframe(test_dir: str) -> pl.DataFrame:
    """Convert test directory structure to DataFrame."""
    records = []
    test_dir = Path(test_dir)

    for case_dir in sorted(test_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        case_id = int(case_dir.name.replace("case", ""))

        for day_dir in sorted(case_dir.iterdir()):
            if not day_dir.is_dir():
                continue
            day_id = int(day_dir.name.replace("day", ""))

            for slice_file in sorted(day_dir.iterdir()):
                if not slice_file.suffix.lower() in [".png", ".jpg"]:
                    continue
                slice_num = int(slice_file.stem.replace("slice_", ""))
                for cls in ["large_bowel", "small_bowel", "stomach"]:
                    records.append(
                        {
                            "case": case_id,
                            "day": day_id,
                            "slice": slice_num,
                            "class": cls,
                            "segmentation": "",
                        }
                    )

    df = pl.DataFrame(records)
    return df


def predict_rle_from_loader(
    models,
    dataloader,
    device,
    submission_template,
    threshold: float = 0.5,
):
    """Run inference and generate RLE predictions."""
    if not isinstance(models, (list, tuple)):
        models = [models]

    for m in models:
        m.to(device)
        m.eval()

    classes = ["large_bowel", "small_bowel", "stomach"]

    # Create a copy of the template for this threshold
    result_df = submission_template.clone()

    with torch.no_grad():
        for images, _, case, day, slice in tqdm(
            dataloader, desc=f"Inference (thr={threshold})"
        ):
            images = images.to(device)
            B = images.shape[0]

            probs_list = []
            for m in models:
                logits = m(images)
                probs = torch.sigmoid(logits)
                probs_list.append(probs)

            stacked = torch.stack(probs_list, dim=0)
            avg_probs = stacked.mean(dim=0)

            preds = (avg_probs > threshold).to(torch.uint8).cpu().numpy()

            for i in range(B):
                for cls_idx in range(preds.shape[1]):
                    mask = preds[i, cls_idx]
                    rle = mask_to_rle(mask)
                    id_ = f"case{case[i]}_day{day[i]}_slice_{str(slice[i]).zfill(4)}_class_{classes[cls_idx]}"

                    result_df = result_df.with_columns(
                        pl.when(pl.col("id").eq(id_))
                        .then(pl.lit(rle))
                        .otherwise(pl.col("segmentation"))
                        .alias("segmentation")
                    )

    return result_df


def run_inference_pipeline(
    models,
    test_data_path: str,
    submission_template_path: str,
    device,
    thresholds: list[float] = [0.5],
    batch_size: int = 8,
    two_point_five_d: bool = False,
    depth: int = 3,
    output_dir: str = "submissions",
    model_name: str = "model",
):
    """Run complete inference pipeline with multiple thresholds."""
    print("\n" + "=" * 60)
    print("Starting Inference Pipeline")
    print("=" * 60)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Prepare test data
    print("Preparing test data...")
    test_data_raw = dir_to_dataframe(test_data_path)

    test_dataset = MRISegmentationDataset(
        dataframe=test_data_raw,
        images_path=test_data_path,
        preprocess_fn=preprocess_mri,
        augmentations=get_test_augmentations(),
        two_point_five_d=two_point_five_d,
        depth=depth,
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    submission_template = pl.read_csv(submission_template_path)

    # Run inference for each threshold
    for threshold in thresholds:
        print(f"\nRunning inference with threshold={threshold}")

        df_submission = predict_rle_from_loader(
            models=models,
            dataloader=test_loader,
            device=device,
            submission_template=submission_template,
            threshold=threshold,
        )

        # Save submission
        output_file = output_path / f"{model_name}_thr{threshold}.csv"
        df_submission.write_csv(output_file)
        print(f"Saved submission to: {output_file}")

    print("\n" + "=" * 60)
    print("Inference completed!")
    print("=" * 60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Main training function."""

    # Configuration
    TRAIN_IMAGES_PATH = "./data/hw3_dataset/train"
    FOLDS_NUM = 4
    EPOCHS = 40
    MODEL_NAME = "efficient_net_unetpp_40ep"
    SAVE_MODEL_PATH = "models"
    PATIENCE = 5
    TWO_POINT_FIVE_D = True
    DEPTH = 5
    BATCH_SIZE = 8
    SCHEDULER_TYPE = "step"

    # Set device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load training data
    print("Loading training data...")
    train_data_raw = (
        pl.read_csv(Path(TRAIN_IMAGES_PATH) / "train.csv")
        .with_columns(pl.col("day").cast(pl.Int32), pl.col("slice").cast(pl.Int32))
        .fill_null("")
    )

    # Add is_empty column
    train_data_raw = train_data_raw.with_columns(
        pl.col("segmentation").eq("").alias("is_empty")
    )

    # Create folds
    print("Creating cross-validation folds...")
    skf = StratifiedGroupKFold(n_splits=FOLDS_NUM, shuffle=True, random_state=42)
    train_data_raw_folded = train_data_raw.with_columns(pl.lit(-1).alias("fold"))

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(
            train_data_raw,
            train_data_raw["is_empty"],
            groups=train_data_raw["case"],
        )
    ):
        train_data_raw_folded = train_data_raw_folded.with_columns(
            pl.when(pl.arange(0, len(train_data_raw)).is_in(val_idx))
            .then(fold)
            .otherwise(pl.col("fold"))
            .alias("fold")
        )

    print("Fold distribution:")
    print(
        train_data_raw_folded.group_by(["fold", "is_empty"])
        .agg(pl.count())
        .sort("fold", "is_empty")
    )

    # Start training
    print(f"\nStarting EfficientNet-B2 + UNet++ training:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Folds: {FOLDS_NUM}")
    print(f"  2.5D: {TWO_POINT_FIVE_D}")
    print(f"  Depth: {DEPTH}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Scheduler: {SCHEDULER_TYPE}")
    print(f"  Loss: Combined (Dice + BCE)")

    models = run_training(
        train_data_raw_folded,
        build_efficientnet_unetpp,
        device,
        TRAIN_IMAGES_PATH,
        folds_num=FOLDS_NUM,
        epochs=EPOCHS,
        model_name=MODEL_NAME,
        save_model_path=SAVE_MODEL_PATH,
        patience=PATIENCE,
        two_point_five_d=TWO_POINT_FIVE_D,
        depth=DEPTH,
        batch_size=BATCH_SIZE,
        scheduler_type=SCHEDULER_TYPE,
    )

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Models saved to: {Path(SAVE_MODEL_PATH) / MODEL_NAME}")
    print("=" * 60)

    # Run inference on test set with multiple thresholds
    TEST_DATA_PATH = "./data/hw3_dataset/test"
    SUBMISSION_TEMPLATE_PATH = "./data/hw3_dataset/sample_submission.csv"
    THRESHOLDS = [0.5, 0.6, 0.7]
    OUTPUT_DIR = "submissions"

    run_inference_pipeline(
        models=models,
        test_data_path=TEST_DATA_PATH,
        submission_template_path=SUBMISSION_TEMPLATE_PATH,
        device=device,
        thresholds=THRESHOLDS,
        batch_size=BATCH_SIZE,
        two_point_five_d=TWO_POINT_FIVE_D,
        depth=DEPTH,
        output_dir=OUTPUT_DIR,
        model_name=MODEL_NAME,
    )


if __name__ == "__main__":
    main()
