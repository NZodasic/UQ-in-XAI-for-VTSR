import os
import glob
import argparse

def compute_dataset_stats(dataset_path):
    import numpy as np
    from PIL import Image
    from tqdm import tqdm

    print(f"Computing statistics for dataset at: {dataset_path}")
    
    # Check if paths exist
    train_images_dir = os.path.join(dataset_path, "images", "train")
    if not os.path.exists(train_images_dir):
        # Fallback to the dataset_path itself
        train_images_dir = dataset_path
        
    image_paths = glob.glob(os.path.join(train_images_dir, "**", "*.jpg"), recursive=True) + \
                  glob.glob(os.path.join(train_images_dir, "**", "*.png"), recursive=True)
                  
    if not image_paths:
        print(f"No images found in {train_images_dir}")
        return None, None
        
    print(f"Found {len(image_paths)} images. Calculating...")
    
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float64) / 255.0
            pixels = img.reshape(-1, 3)
            channel_sum += pixels.sum(axis=0)
            channel_sum_sq += np.square(pixels).sum(axis=0)
            pixel_count += pixels.shape[0]
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    if pixel_count == 0:
        print("No readable images found.")
        return None, None

    mean_rgb = channel_sum / pixel_count
    variance_rgb = channel_sum_sq / pixel_count - np.square(mean_rgb)
    std_rgb = np.sqrt(np.maximum(variance_rgb, 0.0))
    
    print("\n==================================")
    print("Dataset Statistics (RGB):")
    print(f"MEAN: [{mean_rgb[0]:.4f}, {mean_rgb[1]:.4f}, {mean_rgb[2]:.4f}]")
    print(f"STD:  [{std_rgb[0]:.4f}, {std_rgb[1]:.4f}, {std_rgb[2]:.4f}]")
    print("==================================")
    
    return mean_rgb, std_rgb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate RGB mean/std for dataset images.")
    parser.add_argument("dataset_path", nargs="?", default=os.environ.get("VTSR_DATASET_DIR", "dataset"))
    args = parser.parse_args()
    compute_dataset_stats(args.dataset_path)
