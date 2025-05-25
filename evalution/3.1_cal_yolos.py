from ultralytics import YOLO

# Load a model
model = YOLO("./ckpt/yolov11m.pt")

# Validate with a custom dataset
metrics = model.val(data="./ckpt/main_yolov11m.yaml")

print('mAP: ', metrics.box.map)  # map50-95
print('mAP50: ', metrics.box.map50)  # map50
print('mAP75: ', metrics.box.map75)  # map75