import os
from PIL import Image
import shutil

# --------------------- Configuration Parameters ---------------------
CONFIG = {
    "class_map": {
        0: "ship",
        1: "buoy",
        2: "person",
        3: "floating object",
        4: "fixed object"
    },
    "source_folders": [
        "00_gt_img",
        "02_layoutdiff",
        "03_gligen",
        "04_instdiff",
        "05_rc-l2i",
        "06_ours"
    ],
    "label_dir": "./data/labels/labels_ori",
    "image_root": "./data/images",
    "output_root": "./data/boxes",
    "target_size": (224, 224)
}

# --------------------- Directory Management ---------------------
def setup_output_directory(config: dict) -> None:
    """Initialize the output directory structure"""
    if os.path.exists(config["output_root"]):
        shutil.rmtree(config["output_root"])
    
    for folder in config["source_folders"]:
        for class_name in config["class_map"].values():
            os.makedirs(
                os.path.join(config["output_root"], folder, class_name),
                exist_ok=True
            )

# --------------------- Image Processing ---------------------
def parse_label_line(line: str) -> tuple:
    """Parse a single line from label file"""
    parts = list(map(float, line.strip().split()))
    if len(parts) != 5:
        raise ValueError(f"Invalid label line format: {line}")
    return int(parts[0]), parts[1], parts[2], parts[3], parts[4]

def calculate_bbox_coordinates(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    img_width: int,
    img_height: int
) -> tuple:
    """Convert normalized coordinates to absolute pixel values"""
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    width_abs = width * img_width
    height_abs = height * img_height
    
    x0 = max(0, x_center_abs - width_abs/2)
    y0 = max(0, y_center_abs - height_abs/2)
    x1 = min(img_width, x_center_abs + width_abs/2)
    y1 = min(img_height, y_center_abs + height_abs/2)
    
    return x0, y0, x1, y1

def process_single_image(
    img_path: str,
    label_path: str,
    source_folder: str,
    config: dict
) -> None:
    """Process a single image and its corresponding label file"""
    try:
        img = Image.open(img_path)
    except Exception as e:
        print(f"Failed to open image {img_path}: {e}")
        return

    with open(label_path, 'r') as f:
        lines = f.readlines()[1:]  # Skip header line

    for idx, line in enumerate(lines, 1):
        try:
            class_id, x_center, y_center, width, height = parse_label_line(line)
            
            if class_id not in config["class_map"]:
                print(f"Invalid class ID {class_id} in file {label_path}")
                continue
                
            bbox = calculate_bbox_coordinates(
                x_center, y_center, width, height,
                *img.size
            )
            
            crop_img = img.crop(bbox).resize(config["target_size"])
            
            output_path = os.path.join(
                config["output_root"],
                source_folder,
                config["class_map"][class_id],
                f"{os.path.splitext(os.path.basename(img_path))[0]}_{idx:02d}.jpg"
            )
            
            crop_img.save(output_path)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

# --------------------- Main Processing Loop ---------------------
def process_all_images(config: dict) -> None:
    """Process all images in all source folders"""
    for folder in config["source_folders"]:
        src_path = os.path.join(config["image_root"], folder)
        
        for img_file in os.listdir(src_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg')):
                continue
            
            base_name = os.path.splitext(img_file)[0]
            label_path = os.path.join(config["label_dir"], f"{base_name}.txt")
            
            if not os.path.exists(label_path):
                print(f"Skipping image without label: {img_file}")
                continue
            
            process_single_image(
                os.path.join(src_path, img_file),
                label_path,
                folder,
                config
            )

# --------------------- Main Execution ---------------------
def print_directory_structure(config: dict) -> None:
    """Print the output directory structure"""
    print("Processing complete! Output directory structure:")
    print(f"├── {config['output_root']}")
    for folder in config["source_folders"]:
        print(f"│   ├── {folder}")
        for class_name in config["class_map"].values():
            print(f"│   │   ├── {class_name}")

if __name__ == "__main__":
    setup_output_directory(CONFIG)
    process_all_images(CONFIG)
    print_directory_structure(CONFIG)