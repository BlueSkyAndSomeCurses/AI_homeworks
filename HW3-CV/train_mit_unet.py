"""
MiT-B3 encoder + UNet decoder training and inference script.
Copies the MiT + UNet workflow from the ai-hw3-cv.ipynb notebook into
an executable Python module that trains all folds, runs inference,
and writes the submission CSV.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import polars as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import ttach

try:  # Optional wandb logging just like in the notebook
    import wandb
except ImportError:  # pragma: no cover - wandb is optional
    wandb = None


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


# ---------------------------------------------------------------------------
# Global configuration copied from the notebook section
# ---------------------------------------------------------------------------
TRAIN_IMAGES_PATH = Path("./data/hw3_dataset/train")
TEST_DATA_PATH = Path("./data/hw3_dataset/test")
SAMPLE_SUBMISSION_PATH = Path("./data/hw3_dataset/sample_submission.csv")
MODEL_SAVE_ROOT = Path("models")
SUBMISSION_DIR = Path("submissions")
MODEL_NAME = "mit_unet_2.5D_depth5_40eps"
SUBMISSION_FILENAME = "submission_mit_unet_2.5D_depth5_40eps_0.69thr.csv"
FOLDS_NUM = 4
EPOCHS = 40
DEPTH = 5
BATCH_SIZE = 8
PATIENCE = 5
SCHEDULER_TYPE = "onecycle"
THRESHOLD = 0.69
TWO_POINT_FIVE_D = True

train_images_path = str(TRAIN_IMAGES_PATH)
test_data_path = str(TEST_DATA_PATH)
MODEL_SAVE_ROOT.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Utility functions copied from notebook cells
# ---------------------------------------------------------------------------


def rle_to_mask(rle: str, height: int, width: int) -> np.ndarray:
    s = list(map(int, rle.split()))
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((height, width))


def mask_to_rle(mask: np.ndarray) -> str:
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def get_image_and_masks(
    data: pl.DataFrame, case: int, day: int, slice_idx: int, images_path: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image_masks = (
        data.filter(
            pl.col("case").eq(case)
            & pl.col("day").eq(day)
            & pl.col("slice").eq(slice_idx)
        )
        .group_by("case", "day", "slice")
        .agg(pl.col("class"), pl.col("segmentation"))
        .explode("class", "segmentation")
    )

    image = np.array(
        Image.open(f"{images_path}/case{case}/day{day}/slice_{slice_idx}.png"),
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
    image = image_uint16.astype(np.float32)
    image /= max(image.max(), 1.0)
    return image


def get_train_augmentations(img_size: tuple[int, int] = (288, 288)) -> A.Compose:
    return A.Compose(
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


def get_improved_train_augmentations(
    img_size: tuple[int, int] = (288, 288),
) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=max(img_size)),
            A.PadIfNeeded(*img_size, border_mode=cv2.BORDER_CONSTANT, fill=0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=1),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5
            ),
            A.OneOf(
                [
                    A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                    A.ElasticTransform(alpha=1, sigma=50, p=1.0),
                ],
                p=0.25,
            ),
            A.RandomCrop(*img_size),
        ]
    )


def get_test_augmentations(img_size: tuple[int, int] = (288, 288)) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=max(img_size)),
            A.PadIfNeeded(*img_size, border_mode=cv2.BORDER_CONSTANT, fill=0),
        ]
    )


class MRISegmentationDataset(Dataset):
    def __init__(
        self,
        dataframe: pl.DataFrame,
        images_path: str,
        preprocess_fn: Callable | None = None,
        augmentations: Callable | None = None,
        two_point_five_d: bool = False,
        depth: int = 3,
        return_original_shape: bool = False,
    ):
        self.dataframe = dataframe
        self.unique_samples = dataframe.select("case", "day", "slice").unique()
        self.preprocess_fn = preprocess_fn
        self.augmentations = augmentations
        self.images_path = images_path
        self.two_point_five_d = two_point_five_d
        self.depth = depth if depth % 2 == 1 else depth + 1
        self.return_original_shape = return_original_shape

    def __len__(self) -> int:
        return len(self.unique_samples)

    def __getitem__(self, idx: int):
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

        replay_data = None
        if self.augmentations:
            if self.two_point_five_d:
                image_hw_c = np.transpose(image, (1, 2, 0))
                augmented = self.augmentations(image=image_hw_c, mask=mask)
                image = np.transpose(augmented["image"], (2, 0, 1))
                mask = augmented["mask"]
                replay_data = augmented.get("replay", None)
            else:
                augmented = self.augmentations(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
                replay_data = augmented.get("replay", None)

        mask = np.transpose(mask, (2, 0, 1))

        if image.ndim == 2:
            image = np.expand_dims(image, 0)

        image_tensor = torch.tensor(image, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)

        if self.return_original_shape:
            return image_tensor, mask_tensor, case, day, slice_idx, replay_data
        return image_tensor, mask_tensor, case, day, slice_idx

    def _get_2p5d_stack(self, case: int, day: int, center_slice_idx: int) -> np.ndarray:
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
        center_pos = all_slices.index(center_slice_idx)
        idxs = [
            all_slices[np.clip(center_pos + offset, 0, len(all_slices) - 1)]
            for offset in range(-half, half + 1)
        ]

        images = []
        for slice_id in idxs:
            im, _, _, _ = get_image_and_masks(
                self.dataframe, case, day, slice_id, self.images_path
            )
            images.append(im)

        return np.stack(images, axis=0).astype(np.float32)


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = smp.losses.DiceLoss(mode="multilabel")
        self.bce_loss = smp.losses.SoftBCEWithLogitsLoss()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(outputs, targets)
        bce = self.bce_loss(outputs, targets)
        return self.dice_weight * dice + self.bce_weight * bce


def dice_coef(y_true, y_pred, thr: float = 0.5, dim=(2, 3), epsilon: float = 0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=(1, 0))
    return dice


def iou_coef(y_true, y_pred, thr: float = 0.5, dim=(2, 3), epsilon: float = 0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)
    iou = ((inter + epsilon) / (union + epsilon)).mean(dim=(1, 0))
    return iou


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scheduler=None,
) -> float:
    model.train()
    epoch_loss = 0.0
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
    return epoch_loss / max(len(train_loader), 1)


def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    thr: float = 0.5,
):
    model.eval()
    dices, ious = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val   "):
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
    def __init__(
        self,
        patience: int = 5,
        mode: str = "max",
        delta: float = 1e-4,
        save_path: Path | None = None,
    ):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.save_path = Path(save_path) if save_path else None

    def __call__(self, current_score: float, model: torch.nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            self._save_model(model)
            return False

        improvement = (
            current_score > self.best_score + self.delta
            if self.mode == "max"
            else current_score < self.best_score - self.delta
        )

        if improvement:
            self.best_score = current_score
            self.counter = 0
            self._save_model(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def _save_model(self, model: torch.nn.Module) -> None:
        if self.save_path:
            torch.save(model.state_dict(), self.save_path)
            if wandb is not None and wandb.run is not None:
                art = wandb.Artifact(f"model-{wandb.run.name}", type="model")
                art.add_file(str(self.save_path))
                wandb.log_artifact(art)


def get_data_loaders(
    full_dataset: pl.DataFrame,
    fold: int,
    two_point_five_d: bool = False,
    depth: int = 3,
    batch_size: int = 8,
    augmentations: Callable = get_train_augmentations,
) -> tuple[DataLoader, DataLoader]:
    train_dataframe = full_dataset.filter(pl.col("fold").ne(fold))
    val_dataframe = full_dataset.filter(pl.col("fold").eq(fold))

    train_data = MRISegmentationDataset(
        train_dataframe,
        images_path=train_images_path,
        preprocess_fn=preprocess_mri,
        augmentations=augmentations(),
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

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def run_training(
    full_data: pl.DataFrame,
    model_builder: Callable[[torch.device], tuple[list, list, list]],
    device: torch.device,
    augmentations: Callable = get_train_augmentations,
    epochs: int = 50,
    save_model_path: str = "models",
    model_name: str = "undefined",
    patience: int = 5,
    two_point_five_d: bool = False,
    depth: int = 3,
    batch_size: int = 8,
    scheduler_type: str = "onecycle",
) -> list:
    models, loss_fns, optimizers = model_builder(device)
    save_dir = Path(save_model_path) / model_name
    save_dir.mkdir(exist_ok=True, parents=True)

    for fold in range(FOLDS_NUM):
        run = None
        if wandb is not None:
            run = wandb.init(
                project="AI_HW3_CV_MRI_segmentation",
                name=f"{model_name}_fold{fold}",
                group=model_name,
                config={
                    "model_name": model_name,
                    "fold": fold,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "patience": patience,
                    "scheduler": scheduler_type,
                },
            )

        print(f"Training fold {fold}")
        model = models[fold]
        loss_fn = loss_fns[fold]
        optimizer = optimizers[fold]
        if run is not None and wandb is not None:
            wandb.watch(model, log="all", log_freq=100)

        train_loader, val_loader = get_data_loaders(
            full_data,
            fold,
            two_point_five_d=two_point_five_d,
            depth=depth,
            batch_size=batch_size,
            augmentations=augmentations,
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

        if run is not None:
            run.finish()

    return models


def dir_to_dataframe(test_dir: str) -> pl.DataFrame:
    records = []
    base = Path(test_dir)
    for case_dir in sorted(base.iterdir()):
        if not case_dir.is_dir():
            continue
        case_id = int(case_dir.name.replace("case", ""))
        for day_dir in sorted(case_dir.iterdir()):
            if not day_dir.is_dir():
                continue
            day_id = int(day_dir.name.replace("day", ""))
            for slice_file in sorted(day_dir.iterdir()):
                if slice_file.suffix.lower() not in {".png", ".jpg"}:
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
    return pl.DataFrame(records)


def segmentation_collate(batch):
    images, masks, cases, days, slice_idxs, replays = zip(*batch)
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    return images, masks, cases, days, slice_idxs, list(replays)


def post_process_mask(
    pred_mask: np.ndarray, original_size: tuple[int, int], input_size=(288, 288)
):
    H, W = original_size
    mask_original = cv2.resize(pred_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    return mask_original


def predict_rle_from_loader(
    models: list,
    dataframe_raw: pl.DataFrame,
    dataloader: DataLoader,
    device: torch.device,
    submission_template: pl.DataFrame,
    threshold: float = 0.5,
    tta: bool = True,
) -> pl.DataFrame:
    if not isinstance(models, (list, tuple)):
        models = [models]

    wrapped_models = []
    for model in models:
        model.to(device)
        model.eval()
        if tta:
            wrapped_models.append(
                ttach.SegmentationTTAWrapper(
                    model, ttach.aliases.d4_transform(), merge_mode="mean"
                )
            )
        else:
            wrapped_models.append(model)

    classes = ["large_bowel", "small_bowel", "stomach"]

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Infer "):
            images, _, case, day, slice_idx, _ = batch
            images = images.to(device)
            probs_list = []
            for model in wrapped_models:
                logits = model(images)
                probs = torch.sigmoid(logits)
                probs_list.append(probs)

            avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)
            preds = (avg_probs > threshold).to(torch.uint8).cpu().numpy()

            for i in range(preds.shape[0]):
                for cls_idx in range(preds.shape[1]):
                    mask_padded = preds[i, cls_idx]
                    mask = post_process_mask(mask_padded, (266, 266))
                    rle = mask_to_rle(mask)
                    case_val = case[i]
                    day_val = day[i]
                    slice_val = slice_idx[i]
                    record_id = f"case{case_val}_day{day_val}_slice_{str(slice_val).zfill(4)}_class_{classes[cls_idx]}"
                    submission_template = submission_template.with_columns(
                        pl.when(pl.col("id").eq(record_id))
                        .then(pl.lit(rle))
                        .otherwise(pl.col("segmentation"))
                        .alias("segmentation")
                    )
    return submission_template


def run_inference_pipeline(
    models: list,
    test_images_path: str,
    submission_template_path: str,
    device: torch.device,
    threshold: float = 0.5,
    batch_size: int = 8,
    preprocess_fn: Callable | None = None,
    two_point_five_d: bool = False,
    depth: int = 3,
    tta: bool = True,
) -> pl.DataFrame:
    test_data_raw_local = dir_to_dataframe(test_images_path)
    test_dataset = MRISegmentationDataset(
        dataframe=test_data_raw_local,
        images_path=test_images_path,
        preprocess_fn=preprocess_fn or preprocess_mri,
        augmentations=get_test_augmentations(),
        two_point_five_d=two_point_five_d,
        depth=depth,
        return_original_shape=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=segmentation_collate,
    )
    submission_template = pl.read_csv(submission_template_path)
    return predict_rle_from_loader(
        models=models,
        dataframe_raw=test_data_raw_local,
        dataloader=test_loader,
        device=device,
        submission_template=submission_template,
        threshold=threshold,
        tta=tta,
    )


def build_mit_unet(device: torch.device) -> tuple[list, list, list]:
    ENCODER = "mit_b3"
    ENCODER_WEIGHTS = "imagenet"
    IN_CHANNELS = 5
    NUM_CLASSES = 3
    LR = 6e-5
    WEIGHT_DECAY = 1e-4

    models = [
        smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=IN_CHANNELS,
            classes=NUM_CLASSES,
            activation=None,
            decoder_use_norm="batchnorm",
        ).to(device)
        for _ in range(FOLDS_NUM)
    ]

    loss_fns = [
        CombinedLoss(dice_weight=0.5, bce_weight=0.5).to(device)
        for _ in range(FOLDS_NUM)
    ]
    optimizers = [
        optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        for model in models
    ]
    return models, loss_fns, optimizers


def prepare_folds(train_df: pl.DataFrame) -> pl.DataFrame:
    train_df = train_df.with_columns(pl.col("segmentation").eq("").alias("is_empty"))
    skf = StratifiedGroupKFold(n_splits=FOLDS_NUM, shuffle=True, random_state=42)
    folded = train_df.with_columns(pl.lit(-1).alias("fold"))
    dummy_X = np.zeros(len(train_df))
    y = train_df["is_empty"]
    groups = train_df["case"]

    for fold, (_, val_idx) in enumerate(skf.split(dummy_X, y, groups=groups)):
        folded = folded.with_columns(
            pl.when(pl.arange(0, len(train_df)).is_in(val_idx))
            .then(fold)
            .otherwise(pl.col("fold"))
            .alias("fold")
        )
    return folded


def main() -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    train_csv_path = TRAIN_IMAGES_PATH / "train.csv"
    if not train_csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found at {train_csv_path}")

    train_data_raw = (
        pl.read_csv(train_csv_path)
        .with_columns(pl.col("day").cast(pl.Int32), pl.col("slice").cast(pl.Int32))
        .fill_null("")
    )

    train_data_raw_folded = prepare_folds(train_data_raw)
    print("Fold distribution:")
    print(
        train_data_raw_folded.group_by(["fold", "is_empty"])
        .agg(pl.count())
        .sort("fold", "is_empty")
    )

    print("\nStarting MiT + UNet training...")
    models_mit_unet_25d = run_training(
        train_data_raw_folded,
        build_mit_unet,
        device,
        augmentations=get_improved_train_augmentations,
        epochs=EPOCHS,
        model_name=MODEL_NAME,
        save_model_path=str(MODEL_SAVE_ROOT),
        patience=PATIENCE,
        two_point_five_d=TWO_POINT_FIVE_D,
        depth=DEPTH,
        batch_size=BATCH_SIZE,
        scheduler_type=SCHEDULER_TYPE,
    )


    print("\nRunning inference for submission...")
    submission_df = run_inference_pipeline(
        models=models_mit_unet_25d,
        test_images_path=test_data_path,
        submission_template_path=str(SAMPLE_SUBMISSION_PATH),
        device=device,
        threshold=THRESHOLD,
        batch_size=BATCH_SIZE,
        preprocess_fn=preprocess_mri,
        two_point_five_d=TWO_POINT_FIVE_D,
        depth=DEPTH,
        tta=True,
    )

    submission_path = SUBMISSION_DIR / SUBMISSION_FILENAME
    submission_df.write_csv(submission_path)
    print(f"Submission saved to {submission_path}")

    if wandb is not None and wandb.run is not None:
        artifact = wandb.Artifact("submission-file", type="inference")
        artifact.add_file(str(submission_path))
        wandb.log_artifact(artifact)


if __name__ == "__main__":
    main()
