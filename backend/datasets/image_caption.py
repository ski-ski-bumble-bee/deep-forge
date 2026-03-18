"""
Dataset with aspect ratio bucketing.

Handles mixed-resolution datasets by grouping images into resolution buckets
and batching within buckets. Supports image+caption pairs (image1.png, image1.txt).
"""

import os
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from PIL import Image


# Standard bucket resolutions (width, height)
# Covers common aspect ratios at various pixel counts
DEFAULT_BUCKETS = [
    # Square
    (512, 512), (768, 768), (1024, 1024),
    # Landscape
    (640, 512), (768, 512), (832, 512),
    (768, 576), (1024, 768), (1152, 768),
    (1024, 576), (1280, 720), (1344, 768),
    (1152, 896), (1216, 832),
    # Portrait (mirrors of landscape)
    (512, 640), (512, 768), (512, 832),
    (576, 768), (768, 1024), (768, 1152),
    (576, 1024), (720, 1280), (768, 1344),
    (896, 1152), (832, 1216),
]


def find_closest_bucket(
    width: int, height: int, buckets: List[Tuple[int, int]]
) -> Tuple[int, int]:
    """Find the bucket that best matches the image's aspect ratio and size."""
    img_aspect = width / height
    best_bucket = buckets[0]
    best_error = float('inf')

    for bw, bh in buckets:
        bucket_aspect = bw / bh
        # Weighted: prefer matching aspect ratio, then minimize pixel count difference
        aspect_error = abs(img_aspect - bucket_aspect)
        pixel_error = abs((width * height) - (bw * bh)) / (width * height)
        error = aspect_error * 2.0 + pixel_error
        if error < best_error:
            best_error = error
            best_bucket = (bw, bh)

    return best_bucket


class ImageCaptionDataset(Dataset):
    """
    Dataset for image-caption pairs.

    Expects a folder structure:
        dataset_dir/
            image1.png
            image1.txt
            image2.jpg
            image2.txt

    Each .txt contains the caption for the corresponding image.
    Supports: .png, .jpg, .jpeg, .webp, .bmp
    """

    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

    def __init__(
        self,
        dataset_dir: str,
        buckets: Optional[List[Tuple[int, int]]] = None,
        default_caption: str = "",
        center_crop: bool = True,
        random_flip: float = 0.0,
        token_padding_length: Optional[int] = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.buckets = buckets or DEFAULT_BUCKETS
        self.default_caption = default_caption
        self.center_crop = center_crop
        self.random_flip = random_flip

        # Discover all image-caption pairs
        self.samples: List[Dict[str, Any]] = []
        self.bucket_indices: Dict[Tuple[int, int], List[int]] = defaultdict(list)

        self._scan_directory()
        self._assign_buckets()

        print(f"[Dataset] Found {len(self.samples)} samples in {dataset_dir}")
        print(f"[Dataset] Bucket distribution:")
        for bucket, indices in sorted(self.bucket_indices.items()):
            print(f"  {bucket[0]}x{bucket[1]}: {len(indices)} images")

    def _scan_directory(self):
        """Scan directory for image files and their captions."""
        image_files = sorted([
            f for f in self.dataset_dir.iterdir()
            if f.suffix.lower() in self.IMAGE_EXTENSIONS
        ])

        for img_path in image_files:
            caption_path = img_path.with_suffix('.txt')
            caption = self.default_caption
            if caption_path.exists():
                caption = caption_path.read_text(encoding='utf-8').strip()

            # Get image dimensions without fully loading
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"[Dataset] Skipping {img_path}: {e}")
                continue

            self.samples.append({
                'image_path': str(img_path),
                'caption': caption,
                'original_width': width,
                'original_height': height,
            })

    def _assign_buckets(self):
        """Assign each image to its closest resolution bucket."""
        for idx, sample in enumerate(self.samples):
            bucket = find_closest_bucket(
                sample['original_width'],
                sample['original_height'],
                self.buckets,
            )
            sample['bucket'] = bucket
            self.bucket_indices[bucket].append(idx)

    def _get_transform(self, target_w: int, target_h: int) -> transforms.Compose:
        """Build transform pipeline for a given bucket resolution."""
        transform_list = []

        # Resize maintaining aspect ratio, then crop
        if self.center_crop:
            transform_list.extend([
                transforms.Resize(
                    max(target_w, target_h),
                    interpolation=transforms.InterpolationMode.LANCZOS,
                ),
                transforms.CenterCrop((target_h, target_w)),
            ])
        else:
            transform_list.append(
                transforms.Resize(
                    (target_h, target_w),
                    interpolation=transforms.InterpolationMode.LANCZOS,
                )
            )

        if self.random_flip > 0:
            transform_list.append(transforms.RandomHorizontalFlip(p=self.random_flip))

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Scale to [-1, 1]
        ])

        return transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        bucket_w, bucket_h = sample['bucket']

        # Load and transform image
        image = Image.open(sample['image_path']).convert('RGB')
        transform = self._get_transform(bucket_w, bucket_h)
        pixel_values = transform(image)

        return {
            'pixel_values': pixel_values,
            'caption': sample['caption'],
            'bucket': sample['bucket'],
            'original_size': (sample['original_width'], sample['original_height']),
            'target_size': (bucket_w, bucket_h),
        }


class BucketSampler(Sampler):
    """
    Sampler that groups samples by bucket for efficient batching.
    Within each bucket, samples are shuffled.
    Batches never mix buckets (all images in a batch have the same resolution).
    """

    def __init__(
        self,
        dataset: ImageCaptionDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.bucket_indices = dataset.bucket_indices

    def __iter__(self):
        batches = []

        for bucket, indices in self.bucket_indices.items():
            idx_list = list(indices)
            if self.shuffle:
                random.shuffle(idx_list)

            # Create batches from this bucket
            for i in range(0, len(idx_list), self.batch_size):
                batch = idx_list[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        # Shuffle batch order
        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield from batch

    def __len__(self):
        total = 0
        for indices in self.bucket_indices.values():
            n = len(indices)
            if self.drop_last:
                total += (n // self.batch_size) * self.batch_size
            else:
                total += n
        return total


def create_dataloader(
    dataset_dir: str,
    batch_size: int = 1,
    buckets: Optional[List[Tuple[int, int]]] = None,
    shuffle: bool = True,
    num_workers: int = 4,
    drop_last: bool = True,
    center_crop: bool = True,
    random_flip: float = 0.0,
) -> Tuple[DataLoader, ImageCaptionDataset]:
    """Create a DataLoader with bucket sampling."""
    dataset = ImageCaptionDataset(
        dataset_dir=dataset_dir,
        buckets=buckets,
        center_crop=center_crop,
        random_flip=random_flip,
    )

    sampler = BucketSampler(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=bucket_collate_fn,
    )

    return dataloader, dataset


def bucket_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate that handles bucket metadata."""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    captions = [item['caption'] for item in batch]
    buckets = [item['bucket'] for item in batch]
    original_sizes = [item['original_size'] for item in batch]
    target_sizes = [item['target_size'] for item in batch]

    return {
        'pixel_values': pixel_values,
        'captions': captions,
        'buckets': buckets,
        'original_sizes': original_sizes,
        'target_sizes': target_sizes,
    }
