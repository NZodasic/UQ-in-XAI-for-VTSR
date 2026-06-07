import os
import glob
import pickle
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from data.augmentations import get_train_transforms, get_val_transforms

class VietnameseTrafficSignDataset(Dataset):
    def __init__(
        self,
        images_dir,
        labels_dir,
        transform=None,
        logger=None,
        split_name="dataset",
        index_workers=1,
        cache_index=False
    ):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.logger = logger
        self.split_name = split_name
        self.index_workers = max(1, int(index_workers))
        self.cache_index = cache_index
        self.samples = []

        self._load_dataset()

    def _log(self, message):
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def _cache_path(self):
        cache_name = f".{self.split_name}_samples_cache.pkl"
        return os.path.join(self.labels_dir, cache_name)

    def _cached_samples_are_valid(self, samples):
        if not isinstance(samples, list) or not samples:
            return False

        # The cache stores image paths. If the project/dataset was moved, those
        # paths can become stale and would otherwise produce black fallback crops.
        images_root = os.path.realpath(self.images_dir)
        image_paths = {sample[0] for sample in samples if len(sample) >= 1}
        if not image_paths:
            return False

        for img_path in image_paths:
            real_img_path = os.path.realpath(img_path)
            try:
                if os.path.commonpath([images_root, real_img_path]) != images_root:
                    return False
            except ValueError:
                return False
            if not os.path.exists(real_img_path):
                return False

        return True

    def _parse_image_labels(self, img_path):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.labels_dir, basename + '.txt')

        if not os.path.exists(label_path):
            return []

        samples = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            # Handle class indices that are represented as float strings (e.g. '23.0')
                            class_id = int(float(parts[0]))
                            cx, cy, w, h = map(float, parts[1:5])
                            samples.append((img_path, class_id, cx, cy, w, h))
                        except ValueError:
                            continue
        except OSError:
            return []

        return samples

    def _load_dataset(self):
        self._log(f"[{self.split_name}] Scanning images: {self.images_dir}")
        if not os.path.exists(self.images_dir) or not os.path.exists(self.labels_dir):
            self._log(f"[{self.split_name}] Warning: directory not found. Images: {self.images_dir}, Labels: {self.labels_dir}")
            return

        cache_path = self._cache_path()
        if self.cache_index and os.path.exists(cache_path):
            try:
                self._log(f"[{self.split_name}] Loading cached index: {cache_path}")
                with open(cache_path, 'rb') as f:
                    cached_samples = pickle.load(f)
                if not self._cached_samples_are_valid(cached_samples):
                    self._log(f"[{self.split_name}] Cached index has stale image paths, rebuilding.")
                    self.samples = []
                else:
                    self.samples = cached_samples
                    self._log(f"[{self.split_name}] Loaded {len(self.samples)} cached samples.")
                    return
            except (OSError, pickle.PickleError, EOFError) as exc:
                self._log(f"[{self.split_name}] Cache unreadable, rebuilding index: {exc}")
            
        image_extensions = ('*.jpg', '*.jpeg', '*.png')
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(self.images_dir, ext)))

        self._log(
            f"[{self.split_name}] Found {len(image_paths)} images. "
            f"Loading labels with {self.index_workers} workers..."
        )

        if self.index_workers == 1:
            iterator = map(self._parse_image_labels, image_paths)
            for parsed_samples in tqdm(iterator, total=len(image_paths), desc=f"Loading {self.split_name}", unit="img"):
                self.samples.extend(parsed_samples)
        else:
            with ThreadPoolExecutor(max_workers=self.index_workers) as executor:
                iterator = executor.map(self._parse_image_labels, image_paths)
                for parsed_samples in tqdm(iterator, total=len(image_paths), desc=f"Loading {self.split_name}", unit="img"):
                    self.samples.extend(parsed_samples)

        self._log(f"[{self.split_name}] Loaded {len(self.samples)} labeled samples.")

        if self.cache_index:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.samples, f)
                self._log(f"[{self.split_name}] Saved index cache: {cache_path}")
            except OSError as exc:
                self._log(f"[{self.split_name}] Could not save index cache: {exc}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_id, cx, cy, w, h = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
            img_w, img_h = image.size
            
            # Convert YOLO normalized coordinates to absolute pixel coordinates
            abs_w = w * img_w
            abs_h = h * img_h
            abs_cx = cx * img_w
            abs_cy = cy * img_h
            
            x1 = max(0, int(abs_cx - abs_w / 2))
            y1 = max(0, int(abs_cy - abs_h / 2))
            x2 = min(img_w, int(abs_cx + abs_w / 2))
            y2 = min(img_h, int(abs_cy + abs_h / 2))
            
            # Bounding box sanity check
            if x2 <= x1 or y2 <= y1:
                crop = image  # Fallback to whole image
            else:
                crop = image.crop((x1, y1, x2, y2))
                
        except Exception as exc:
            raise RuntimeError(
                f"[{self.split_name}] Failed to load image crop from {img_path}"
            ) from exc
            
        if self.transform:
            crop = self.transform(crop)
            
        return crop, class_id

def get_dataloaders(config, logger=None):
    data_cfg = config['data']
    
    img_size = tuple(data_cfg.get('image_size', [224, 224]))
    batch_size = data_cfg.get('batch_size', 32)
    num_workers = max(0, int(data_cfg.get('num_workers', 4)))
    pin_memory = data_cfg.get('pin_memory', torch.cuda.is_available())
    index_workers = data_cfg.get('index_workers', 1)
    cache_index = data_cfg.get('cache_index', False)
    prefetch_factor = int(data_cfg.get('prefetch_factor', 2))
    dataloader_kwargs = {
        'num_workers': num_workers,
        'persistent_workers': num_workers > 0,
        'pin_memory': pin_memory
    }
    if num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = max(1, prefetch_factor)
    
    train_transform = get_train_transforms(img_size)
    val_transform = get_val_transforms(img_size)
    test_transform = get_val_transforms(img_size)
    
    train_dataset = VietnameseTrafficSignDataset(
        data_cfg.get('train_images_dir', ''), 
        data_cfg.get('train_labels_dir', ''), 
        transform=train_transform,
        logger=logger,
        split_name="train",
        index_workers=index_workers,
        cache_index=cache_index
    )

    if len(train_dataset.samples) == 0:
        raise RuntimeError(
            "Training dataset is empty. Please check:\n"
            f"  train_images_dir: {data_cfg.get('train_images_dir', '')}\n"
            f"  train_labels_dir: {data_cfg.get('train_labels_dir', '')}"
        )
    
    val_dataset = VietnameseTrafficSignDataset(
        data_cfg.get('val_images_dir', ''), 
        data_cfg.get('val_labels_dir', ''), 
        transform=val_transform,
        logger=logger,
        split_name="val",
        index_workers=index_workers,
        cache_index=cache_index
    )
    
    test_dataset = VietnameseTrafficSignDataset(
        data_cfg.get('test_images_dir', ''), 
        data_cfg.get('test_labels_dir', ''), 
        transform=test_transform,
        logger=logger,
        split_name="test",
        index_workers=index_workers,
        cache_index=cache_index
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **dataloader_kwargs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **dataloader_kwargs
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **dataloader_kwargs
    )
    
    return train_loader, val_loader, test_loader
