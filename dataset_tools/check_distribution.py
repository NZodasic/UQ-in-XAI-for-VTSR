import os
import argparse
from collections import Counter

def get_class_names(yaml_path):
    import yaml

    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    names = data.get('names', {})
    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}
    return {int(k): v for k, v in names.items()}

def check_distribution(label_dir):
    class_counts = Counter()
    for txt_name in os.listdir(label_dir):
        txt_path = os.path.join(label_dir, txt_name)
        if not txt_name.endswith('.txt'):
            continue
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        class_id = int(float(parts[0]))
                        class_counts[class_id] += 1
        except Exception as e:
            print(f"Error reading {txt_path}: {e}")
    return class_counts

def main():
    parser = argparse.ArgumentParser(description="Check YOLO label class distribution by split.")
    parser.add_argument("base_dir", nargs="?", default=os.environ.get("VTSR_DATASET_DIR", "dataset"))
    parser.add_argument("--classes-file", default=None, help="Path to custom_data.yaml/classes yaml file.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    base_dir = args.base_dir
    yaml_path = args.classes_file or os.path.join(base_dir, "custom_data.yaml")
    
    if not os.path.exists(yaml_path):
        print(f"YAML file not found: {yaml_path}")
        return
        
    class_names = get_class_names(yaml_path)
    print("Class names:", class_names)
    
    for split in args.splits:
        label_dir = os.path.join(base_dir, "labels", split)
        if os.path.exists(label_dir):
            counts = check_distribution(label_dir)
            print(f"\nDistribution for {split} split:")
            total = sum(counts.values())
            print(f"Total annotations: {total}")
            for cls_id in sorted(counts.keys()):
                name = class_names.get(cls_id, f"Class {cls_id}")
                percent = counts[cls_id] / total * 100 if total else 0.0
                print(f"  {cls_id:2d} ({name:30s}): {counts[cls_id]:6d} ({percent:.2f}%)")
        else:
            print(f"\nLabel directory not found for split {split}: {label_dir}")

if __name__ == "__main__":
    main()
