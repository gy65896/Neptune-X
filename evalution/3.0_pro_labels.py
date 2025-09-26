import os
from typing import List

# Configuration parameters
CONFIG = {
    "source_folder": "./data/labels/labels_ori",
    "method_folders": [
        '02_layoutdiff', 
        '03_gligen', 
        '04_instdiff', 
        '05_rc-l2i', 
        '06_ours'
    ],
    "labels_root": "./data/labels"  # Common root path for all label folders
}

def ensure_directories_exist(directories: List[str]) -> None:
    """Create target directories if they don't exist"""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def process_text_file(source_path: str, target_path: str) -> None:
    """Process a single text file by removing first line and saving to target"""
    try:
        with open(source_path, 'r', encoding='utf-8') as src_file:
            lines = src_file.readlines()[1:]  # Skip first line
        
        with open(target_path, 'w', encoding='utf-8') as tgt_file:
            tgt_file.writelines(lines)
    except UnicodeDecodeError:
        print(f"Encoding error in file: {source_path}")
    except IOError as e:
        print(f"File operation failed for {source_path}: {e}")

def process_all_files(config: dict) -> None:
    """Process all text files from source to target folders"""
    # Build full target paths
    target_folders = [os.path.join(config["labels_root"], method) 
                     for method in config["method_folders"]]
    
    ensure_directories_exist(target_folders)
    processed_files = 0

    for filename in os.listdir(config["source_folder"]):
        if filename.endswith(".txt"):
            source_file = os.path.join(config["source_folder"], filename)
            
            for target_folder in target_folders:
                target_file = os.path.join(target_folder, filename)
                process_text_file(source_file, target_file)
                processed_files += 1

    print(f"Processing complete! Modified {processed_files} files across {len(target_folders)} target folders.")

if __name__ == "__main__":
    process_all_files(CONFIG)