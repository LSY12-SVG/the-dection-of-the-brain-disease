import random
from pathlib import Path
import shutil

def split_dataset(base_dir, val_ratio=0.2):
    base_dir = Path(base_dir)
    images_train = base_dir / 'images' / 'train'
    labels_train = base_dir / 'labels' / 'train'
    
    images_val = base_dir / 'images' / 'val'
    labels_val = base_dir / 'labels' / 'val'
    
    images_val.mkdir(parents=True, exist_ok=True)
    labels_val.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    images = list(images_train.glob('*.*'))
    random.shuffle(images)
    
    val_count = int(len(images) * val_ratio)
    val_images = images[:val_count]
    
    print(f"Moving {len(val_images)} images to val...")
    
    for img_path in val_images:
        # Move image
        shutil.move(str(img_path), str(images_val / img_path.name))
        
        # Move corresponding label
        label_name = img_path.stem + '.txt'
        label_src = labels_train / label_name
        if label_src.exists():
            shutil.move(str(label_src), str(labels_val / label_name))

if __name__ == '__main__':
    split_dataset("肿瘤模型数据集/yolo_seg_ds")
