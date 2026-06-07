import argparse
import glob
import os

def check_classes(label_dir):
    classes = set()
    count = 0
    for f in glob.glob(os.path.join(label_dir, '*.txt')):
        count += 1
        with open(f, encoding='utf-8') as fp:
            for line in fp:
                parts = line.strip().split()
                if parts:
                    classes.add(int(float(parts[0])))
    return classes, count

def main():
    parser = argparse.ArgumentParser(description="List unique class ids in a YOLO label directory.")
    parser.add_argument("label_dir", nargs="?", default=os.path.join("dataset", "labels", "train"))
    args = parser.parse_args()

    classes, count = check_classes(args.label_dir)
    print(f"Label directory: {args.label_dir}")
    print(f"Unique classes: {sorted(classes)}")
    print(f"Total unique classes: {len(classes)}")
    print(f"Max class id: {max(classes) if classes else None}")
    print(f"Label files: {count}")

if __name__ == "__main__":
    main()
