import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from typing import List, Dict
import argparse

# 配置参数集中管理
CONFIG = {
    "model_path": "./ckpt/resnet.pth",
    "data_root": "./data/boxes",
    "methods": [
        '02_layoutdiff', 
        '03_gligen', 
        '04_instdiff', 
        '05_rc-l2i', 
        '06_ours'
    ],
    "class_order": ["ship", "buoy", "person", "floating object", "fixed object"],
    "img_size": 224,
    "batch_size": 32,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def create_transform(img_size: int) -> transforms.Compose:

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def load_model(config: Dict) -> nn.Module:

    model = models.resnet101(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(config["class_order"]))
    
    checkpoint = torch.load(config["model_path"], map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    return model.to(config["device"])

def validate_model(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return correct / total

def create_dataset(method: str, config: Dict) -> datasets.ImageFolder:
    transform = create_transform(config["img_size"])
    dataset = datasets.ImageFolder(
        root=f"{config['data_root']}/{method}",
        transform=transform
    )
    
    dataset.classes = sorted(dataset.classes, key=lambda x: config["class_order"].index(x))
    dataset.class_to_idx = {cls: idx for idx, cls in enumerate(dataset.classes)}
    
    return dataset

def main(config: Dict):
    torch.manual_seed(42)

    model = load_model(config)
    print(f"Model loaded from {config['model_path']}")
    
    results = {}
    for method in config["methods"]:
        try:
            dataset = create_dataset(method, config)
            loader = DataLoader(
                dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=config["num_workers"],
                pin_memory=True
            )

            acc = validate_model(model, loader, config["device"])
            results[method] = acc
            print(f"{method}: Accuracy = {acc:.4f}")
        except Exception as e:
            print(f"Error processing {method}: {str(e)}")
            results[method] = None
    
    print("\nFinal Results:")
    for method, acc in results.items():
        print(f"{method}: {acc if acc is not None else 'Failed'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=CONFIG["model_path"], help="Path to model checkpoint")
    parser.add_argument("--data", type=str, default=CONFIG["data_root"], help="Root directory of data")
    parser.add_argument("--methods", nargs="+", default=CONFIG["methods"], help="List of methods to evaluate")
    parser.add_argument("--classes", nargs="+", default=CONFIG["class_order"], help="Class order for sorting")
    parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"], help="Batch size for evaluation")
    args = parser.parse_args()
    
    CONFIG.update({
        "model_path": args.model,
        "data_root": args.data,
        "methods": args.methods,
        "class_order": args.classes,
        "batch_size": args.batch_size
    })
    
    main(CONFIG)
