import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class VietnameseTrafficSignDataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        
        self.samples = []
        if os.path.exists(images_dir) and os.path.exists(labels_dir):
            for f in os.listdir(images_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    label_name = os.path.splitext(f)[0] + '.txt'
                    label_path = os.path.join(self.labels_dir, label_name)
                    
                    if os.path.exists(label_path):
                        with open(label_path, 'r') as file:
                            lines = file.readlines()
                            if lines:
                                # For classifier, use the first label
                                class_idx = int(lines[0].split()[0])
                                self.samples.append((f, class_idx, label_path))
                                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_name, class_idx, label_path = self.samples[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, class_idx, img_path

def get_dataloaders(config: dict):
    from .augmentations import get_train_transforms, get_val_transforms
    
    train_transform = get_train_transforms(config['data']['image_size'])
    val_transform = get_val_transforms(config['data']['image_size'])
    
    train_dataset = VietnameseTrafficSignDataset(
        config['data']['train_images_dir'],
        config['data']['train_labels_dir'],
        transform=train_transform
    )
    
    val_dataset = VietnameseTrafficSignDataset(
        config['data']['val_images_dir'],
        config['data']['val_labels_dir'],
        transform=val_transform
    )
    
    test_dataset = VietnameseTrafficSignDataset(
        config['data']['test_images_dir'],
        config['data']['test_labels_dir'],
        transform=val_transform
    )
    
    from utils.seed import worker_init_fn
    
    g = torch.Generator()
    g.manual_seed(config['experiment']['seed'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        worker_init_fn=worker_init_fn,
        generator=g,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        worker_init_fn=worker_init_fn,
        generator=g
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        worker_init_fn=worker_init_fn,
        generator=g
    )
    
    return train_loader, val_loader, test_loader
