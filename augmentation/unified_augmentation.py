from __future__ import annotations

import os
import sys
import argparse
import random
import math
from typing import Dict, List, Tuple
from collections import Counter
import shutil
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

cv2 = None
np = None
A = None

def load_augmentation_dependencies():
    global cv2, np, A
    if cv2 is None:
        import cv2 as cv2_module
        cv2 = cv2_module
    if np is None:
        import numpy as np_module
        np = np_module
    if A is None:
        import albumentations as albumentations_module
        A = albumentations_module

# Handle Colab environment automatically
IN_COLAB = 'google.colab' in sys.modules or os.path.exists('/content')
if IN_COLAB:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except ImportError:
        pass

class AugmentationConfig:
    if IN_COLAB:
        DEFAULT_BASE_DIR = "/content/drive/MyDrive/TSR-VIP/dataset/data2"
        DEFAULT_OUTPUT_AUGMENTED = "/content/drive/MyDrive/TSR-VIP/dataset/data2-augment"
    else:
        DEFAULT_BASE_DIR = r"E:\TSR\UQ-in-XAI-for-VTSR\Dataset\data2"
        DEFAULT_OUTPUT_AUGMENTED = r"E:\TSR\UQ-in-XAI-for-VTSR\Dataset\data2-augment"

    def __init__(
        self,
        base_dir: str = None,
        output_augmented: str = None,
        classes_file: str = None,
        augmentations_per_image: int = 6,
        max_augmentations_per_image: int = 25,
        min_samples_per_class: int = 150,
        oversampling_factor: float = 2.5,
        random_seed: int = 42,
        num_workers: int = None,
        use_multiprocessing: bool = True
    ):
        self.BASE_DIR = base_dir or os.environ.get("VTSR_DATASET_DIR", self.DEFAULT_BASE_DIR)
        self.OUTPUT_AUGMENTED = output_augmented or os.environ.get("VTSR_AUGMENTED_DIR", self.DEFAULT_OUTPUT_AUGMENTED)
        self.CLASSES_FILE = classes_file or os.path.join(self.BASE_DIR, "custom_data.yaml")
        self.AUGMENTATIONS_PER_IMAGE = max(1, int(augmentations_per_image))
        self.MAX_AUGMENTATIONS_PER_IMAGE = max(self.AUGMENTATIONS_PER_IMAGE, int(max_augmentations_per_image))
        self.MIN_SAMPLES_PER_CLASS = max(1, int(min_samples_per_class))
        self.OVERSAMPLING_FACTOR = float(oversampling_factor)
        self.RANDOM_SEED = int(random_seed)
        self.NUM_WORKERS = max(1, int(num_workers if num_workers is not None else min(cpu_count() - 1, 8)))
        self.USE_MULTIPROCESSING = bool(use_multiprocessing)

def make_transform(transform_cls, primary_kwargs, fallback_kwargs):
    try:
        return transform_cls(**primary_kwargs)
    except TypeError:
        return transform_cls(**fallback_kwargs)

class AdvancedAugmentationPipeline:
    def __init__(self, config: AugmentationConfig):
        load_augmentation_dependencies()
        self.config = config
        random.seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)

        self._pipeline = self._build_advanced_pipeline()

    def _build_advanced_pipeline(self):
        return A.Compose([
            # 1. Geometric & Spatial
            A.SomeOf([
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=1.0, border_mode=cv2.BORDER_CONSTANT),
                A.Affine(shear={'x': (-5, 5), 'y': (-5, 5)}, translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)}, p=1.0),
                A.Perspective(scale=(0.02, 0.06), p=1.0),
            ], n=1, p=0.7),
            
            # 2. Lighting & Color
            A.SomeOf([
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
                A.RandomGamma(gamma_limit=(75, 125), p=1.0),
                A.CLAHE(clip_limit=3.0, p=1.0),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                make_transform(
                    A.RandomSunFlare,
                    {
                        'flare_roi': (0, 0, 1, 0.5),
                        'angle_range': (0, 1),
                        'num_flare_circles_range': (1, 3),
                        'src_radius': 150,
                        'src_color': (255, 255, 255),
                        'p': 1.0
                    },
                    {
                        'flare_roi': (0, 0, 1, 0.5),
                        'angle_lower': 0,
                        'angle_upper': 1,
                        'num_flare_circles_lower': 1,
                        'num_flare_circles_upper': 3,
                        'src_radius': 150,
                        'src_color': (255, 255, 255),
                        'p': 1.0
                    }
                ),
            ], n=2, p=0.8),
            
            # 3. Camera artifacts & Blur
            A.SomeOf([
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                make_transform(
                    A.ImageCompression,
                    {'quality_range': (75, 95), 'p': 1.0},
                    {'quality_lower': 75, 'quality_upper': 95, 'p': 1.0}
                ),
            ], n=1, p=0.5),
            
            # 4. Occlusion & Weather
            A.SomeOf([
                make_transform(
                    A.CoarseDropout,
                    {
                        'num_holes_range': (1, 2),
                        'hole_height_range': (8, 32),
                        'hole_width_range': (8, 32),
                        'fill': 0,
                        'p': 1.0
                    },
                    {
                        'max_holes': 2,
                        'max_height': 32,
                        'max_width': 32,
                        'min_holes': 1,
                        'min_height': 8,
                        'min_width': 8,
                        'fill_value': 0,
                        'p': 1.0
                    }
                ),
                A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.08, p=1.0),
                make_transform(
                    A.RandomRain,
                    {'slant_range': (-10, 10), 'drop_length': 15, 'drop_width': 1, 'p': 1.0},
                    {'slant_lower': -10, 'slant_upper': 10, 'drop_length': 15, 'drop_width': 1, 'p': 1.0}
                ),
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=1.0),
            ], n=1, p=0.4),
            
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=25, min_visibility=0.4))

    def apply_random_pipeline(self, image: np.ndarray, bboxes: List[List[float]], class_labels: List[int]) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        try:
            augmented = self._pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
            return augmented['image'], augmented['bboxes'], augmented['class_labels']
        except Exception:
            # Silent fallback
            return image, bboxes, class_labels

class ClassBalancedSampler:
    def __init__(
        self,
        image_files: List[str],
        label_files: List[str],
        min_samples: int = 100,
        oversampling_factor: float = 2.0,
        default_augmentations: int = 1,
        max_augmentations: int = 25
    ):
        self.image_files = image_files
        self.label_files = label_files
        self.min_samples = min_samples
        self.oversampling_factor = oversampling_factor
        self.default_augmentations = max(1, int(default_augmentations))
        self.max_augmentations = max(self.default_augmentations, int(max_augmentations))
        self.class_counts = self._analyze_distribution()
        self.augmentation_schedule = self._compute_schedule()

    def _analyze_distribution(self) -> Counter:
        class_counts = Counter()
        for label_file in self.label_files:
            if os.path.exists(label_file):
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    for line in lines:
                        items = line.strip().split()
                        if len(items) >= 5:
                            cls = int(float(items[0]))
                            class_counts[cls] += 1
                except (OSError, ValueError):
                    continue
        return class_counts

    def _compute_schedule(self) -> Dict[str, int]:
        if not self.class_counts:
            return {}
            
        schedule = {}
        for label_file in self.label_files:
            if not os.path.exists(label_file):
                schedule[label_file] = 1
                continue
            try:
                with open(label_file, 'r') as f:
                    classes_in_image = [int(float(line.strip().split()[0])) for line in f if len(line.strip().split()) >= 5]

                if not classes_in_image:
                    schedule[label_file] = 1
                    continue

                min_class_count = min(self.class_counts[c] for c in classes_in_image)
                
                # Class-aware augmentation schedule using soft schedule formula
                N_max = max(self.class_counts.values()) if self.class_counts else 1
                alpha = self.oversampling_factor
                
                soft_aug = int(math.ceil(math.sqrt(N_max / max(1, min_class_count)) * alpha))
                target_aug = int(math.ceil(max(0, self.min_samples - min_class_count) / max(1, min_class_count)))
                aug_times = max(self.default_augmentations, soft_aug, target_aug)
                aug_times = min(self.max_augmentations, aug_times)
                schedule[label_file] = max(1, aug_times)
            except (OSError, ValueError):
                schedule[label_file] = 1

        return schedule

    def get_augmentation_count(self, label_file: str) -> int:
        return self.augmentation_schedule.get(label_file, 1)

def load_class_names(yaml_file: str) -> Dict[int, str]:
    import yaml

    if not os.path.exists(yaml_file):
        # Fallback to checking classes.txt 
        txt_file = yaml_file.replace("custom_data.yaml", "classes.txt")
        if os.path.exists(txt_file):
            with open(txt_file, 'r') as f:
                return {i: line.strip() for i, line in enumerate(f)}
        return {i: f"Class {i}" for i in range(29)}
        
    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        if 'names' in data and isinstance(data['names'], dict):
            return {int(k): v for k, v in data['names'].items()}
        elif 'names' in data and isinstance(data['names'], list):
            return {i: name for i, name in enumerate(data['names'])}
    except Exception as e:
        print(f"Warning: Could not load YAML smoothly: {e}")
    return {i: f"Class {i}" for i in range(29)}

def read_yolo_labels(label_file: str) -> Tuple[List[List[float]], List[int]]:
    bboxes, class_labels = [], []
    if not os.path.exists(label_file): return bboxes, class_labels
    try:
        with open(label_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            items = line.strip().split()
            if len(items) >= 5:
                cls = int(float(items[0]))
                bbox = list(map(float, items[1:5]))
                class_labels.append(cls)
                bboxes.append(bbox)
    except (OSError, ValueError):
        pass
    return bboxes, class_labels

def write_yolo_labels(label_file: str, bboxes: List[List[float]], class_labels: List[int]):
    with open(label_file, 'w') as f:
        for cls, bbox in zip(class_labels, bboxes):
            bbox_str = ' '.join([f"{x:.6f}" for x in bbox])
            f.write(f"{cls} {bbox_str}\n")

def validate_yolo_bboxes(bboxes: List[List[float]], class_labels: List[int]) -> Tuple[List[List[float]], List[int]]:
    valid = []
    for bbox, cls in zip(bboxes, class_labels):
        cx, cy, bw, bh = bbox
        if (
            cls >= 0
            and 0 < cx < 1
            and 0 < cy < 1
            and 0.01 < bw <= 1
            and 0.01 < bh <= 1
            and cx - bw / 2 >= 0
            and cx + bw / 2 <= 1
            and cy - bh / 2 >= 0
            and cy + bh / 2 <= 1
        ):
            valid.append((bbox, cls))
    if not valid:
        return [], []
    return [b for b, _ in valid], [c for _, c in valid]

def process_single_image(args):
    load_augmentation_dependencies()
    img_path, label_path, output_img_dir, output_lbl_dir, n_augmentations, random_seed = args
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    config = AugmentationConfig(random_seed=random_seed)
    aug_pipeline = AdvancedAugmentationPipeline(config)
    basename = os.path.splitext(os.path.basename(img_path))[0]

    try:
        image = cv2.imread(img_path)
        if image is None: return 0
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes, class_labels = read_yolo_labels(label_path)
        bboxes, class_labels = validate_yolo_bboxes(bboxes, class_labels)
        if not bboxes: return 0

        # Output original as well using PNG (better quality preservation)
        orig_img_name = f"{basename}.png"
        cv2.imwrite(os.path.join(output_img_dir, orig_img_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])
        write_yolo_labels(os.path.join(output_lbl_dir, f"{basename}.txt"), bboxes, class_labels)
        
        successful = 0
        for aug_idx in range(n_augmentations):
            try:
                aug_img, aug_bboxes, aug_labels = aug_pipeline.apply_random_pipeline(image.copy(), bboxes.copy(), class_labels.copy())
                aug_bboxes, aug_labels = validate_yolo_bboxes(aug_bboxes, aug_labels)
                if not aug_bboxes: continue

                aug_img_name = f"{basename}_aug{aug_idx}.png"
                cv2.imwrite(
                    os.path.join(output_img_dir, aug_img_name),
                    cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_PNG_COMPRESSION, 3]
                )
                write_yolo_labels(os.path.join(output_lbl_dir, f"{basename}_aug{aug_idx}.txt"), aug_bboxes, aug_labels)
                successful += 1
            except Exception:
                continue
        return successful
    except Exception:
        return 0

def validate_output_directory(base_dir: str, output_dir: str):
    base_abs = os.path.abspath(base_dir)
    output_abs = os.path.abspath(output_dir)
    if output_abs == base_abs:
        raise ValueError("Output directory must be different from BASE_DIR.")
    try:
        common_path = os.path.commonpath([base_abs, output_abs])
    except ValueError:
        return
    if common_path == output_abs:
        raise ValueError("Output directory cannot be a parent of BASE_DIR.")

def copy_split_if_exists(src: str, dst: str):
    if os.path.exists(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)

def augment_dataset(config: AugmentationConfig):
    load_augmentation_dependencies()
    import yaml
    from tqdm import tqdm

    validate_output_directory(config.BASE_DIR, config.OUTPUT_AUGMENTED)
    class_names = load_class_names(config.CLASSES_FILE)
    print(f"\nLoaded {len(class_names)} classes")

    # Assuming Dataset is already split into train, val, test inside BASE_DIR
    src_train_img = os.path.join(config.BASE_DIR, 'images', 'train')
    src_train_lbl = os.path.join(config.BASE_DIR, 'labels', 'train')
    
    dst_train_img = os.path.join(config.OUTPUT_AUGMENTED, 'images', 'train')
    dst_train_lbl = os.path.join(config.OUTPUT_AUGMENTED, 'labels', 'train')

    if not os.path.exists(src_train_img):
        print(f"ERROR: Cannot find training images at {src_train_img}")
        print("Please ensure your dataset is already split into images/train, labels/train etc. inside BASE_DIR.")
        return
    if not os.path.exists(src_train_lbl):
        print(f"ERROR: Cannot find training labels at {src_train_lbl}")
        return

    # Clean up output directory
    if os.path.exists(config.OUTPUT_AUGMENTED):
        print(f"Cleaning up previous directory: {config.OUTPUT_AUGMENTED} ...")
        shutil.rmtree(config.OUTPUT_AUGMENTED)

    os.makedirs(dst_train_img, exist_ok=True)
    os.makedirs(dst_train_lbl, exist_ok=True)

    # Copy val and test sets
    print("\nCopying val and test sets...")
    for split in ['val', 'test']:
        s_img = os.path.join(config.BASE_DIR, 'images', split)
        s_lbl = os.path.join(config.BASE_DIR, 'labels', split)
        d_img = os.path.join(config.OUTPUT_AUGMENTED, 'images', split)
        d_lbl = os.path.join(config.OUTPUT_AUGMENTED, 'labels', split)
        
        copy_split_if_exists(s_img, d_img)
        copy_split_if_exists(s_lbl, d_lbl)

    # Handle train set
    all_images = sorted([f for f in os.listdir(src_train_img) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not all_images:
        print(f"ERROR: No training images found at {src_train_img}")
        return

    train_label_files = [os.path.join(src_train_lbl, os.path.splitext(f)[0] + '.txt') for f in all_images]

    sampler = ClassBalancedSampler(
        all_images,
        train_label_files,
        config.MIN_SAMPLES_PER_CLASS,
        config.OVERSAMPLING_FACTOR,
        default_augmentations=config.AUGMENTATIONS_PER_IMAGE,
        max_augmentations=config.MAX_AUGMENTATIONS_PER_IMAGE
    )

    print(f"\nClass distribution (Before Augmentation):")
    for cls, count in sorted(sampler.class_counts.items()):
        class_name = class_names.get(cls, 'Unknown')
        print(f"  Class {cls:2d} ({class_name:30s}): {count:4d} samples")

    total_aug_per_image = sum(sampler.augmentation_schedule.values())
    avg_aug_per_image = total_aug_per_image / len(sampler.augmentation_schedule) if sampler.augmentation_schedule else 0

    print(f"\nAugmentation Strategy:")
    print(f"  Class-aware Smart Sampling Enabled")
    print(f"  Average augmentations target per image: {avg_aug_per_image:.2f}")

    tasks = []
    for img_name in all_images:
        basename = os.path.splitext(img_name)[0]
        img_path = os.path.join(src_train_img, img_name)
        label_path = os.path.join(src_train_lbl, f"{basename}.txt")
        n_augmentations = sampler.get_augmentation_count(label_path)
        
        tasks.append((
            img_path,
            label_path,
            dst_train_img,
            dst_train_lbl,
            n_augmentations,
            config.RANDOM_SEED + len(tasks)
        ))

    print(f"\n--- Creating Augmented Dataset (Multiprocessing) ---")
    if config.USE_MULTIPROCESSING and config.NUM_WORKERS > 1:
        with Pool(processes=config.NUM_WORKERS) as pool:
            results = list(tqdm(pool.imap(process_single_image, tasks), total=len(tasks), desc="Augmenting"))
    else:
        results = [process_single_image(t) for t in tqdm(tasks, desc="Augmenting")]

    total_augmented = sum(results)
    
    # --------------------------
    # Output File Structure Fix
    # --------------------------
    # Copy custom_data.yaml and classes.txt to match user desired structure
    if os.path.exists(config.CLASSES_FILE):
        shutil.copy2(config.CLASSES_FILE, os.path.join(config.OUTPUT_AUGMENTED, "custom_data.yaml"))
    
    classes_txt_src = os.path.join(config.BASE_DIR, "classes.txt")
    if os.path.exists(classes_txt_src):
        shutil.copy2(classes_txt_src, os.path.join(config.OUTPUT_AUGMENTED, "classes.txt"))

    # Also dynamically create a YOLO data.yaml as a bonus configuration
    yaml_content = {
        'train': os.path.join(config.OUTPUT_AUGMENTED, 'images/train').replace('\\', '/'),
        'val': os.path.join(config.OUTPUT_AUGMENTED, 'images/val').replace('\\', '/'),
        'test': os.path.join(config.OUTPUT_AUGMENTED, 'images/test').replace('\\', '/'),
        'nc': len(class_names),
        'names': class_names
    }
    with open(os.path.join(config.OUTPUT_AUGMENTED, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print(f"\n======================================================================")
    print(f"AUGMENTATION COMPLETED")
    print(f"======================================================================")
    print(f"Output saved at: {config.OUTPUT_AUGMENTED}")
    print(f"Original Train Samples: {len(all_images)}")
    print(f"Augmented Train Samples Produced: {total_augmented}")
    print(f"Total Train Samples Now: {len(all_images) + total_augmented}")
    print(f"======================================================================")

    # Check missing
    missing = [i for i in range(len(class_names)) if i not in sampler.class_counts]
    if missing:
        print(f"\n[WARNING] Skipped classes with no source training samples: {missing}")
        print("Use class weights or fix the train split before training if these classes are required.")

def parse_args():
    parser = argparse.ArgumentParser(description="Create a class-balanced augmented YOLO dataset.")
    parser.add_argument("--base-dir", default=None, help="Input dataset root containing images/ and labels/.")
    parser.add_argument("--output-dir", default=None, help="Output dataset root for augmented data.")
    parser.add_argument("--classes-file", default=None, help="Path to custom_data.yaml or equivalent class file.")
    parser.add_argument("--augmentations-per-image", type=int, default=6)
    parser.add_argument("--max-augmentations-per-image", type=int, default=25)
    parser.add_argument("--min-samples-per-class", type=int, default=150)
    parser.add_argument("--oversampling-factor", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-multiprocessing", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = AugmentationConfig(
        base_dir=args.base_dir,
        output_augmented=args.output_dir,
        classes_file=args.classes_file,
        augmentations_per_image=args.augmentations_per_image,
        max_augmentations_per_image=args.max_augmentations_per_image,
        min_samples_per_class=args.min_samples_per_class,
        oversampling_factor=args.oversampling_factor,
        random_seed=args.seed,
        num_workers=args.workers,
        use_multiprocessing=not args.no_multiprocessing
    )
    augment_dataset(config)
