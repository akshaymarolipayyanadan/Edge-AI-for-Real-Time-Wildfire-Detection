import glob
import time
from ultralytics import YOLO

model = YOLO('/home/pi/fire_project/fire_nano_trained_int8.tflite')

images = glob.glob('*.jpg')  # put 10-15 fire and non-fire images in folder

for img in images:
    results = model(img)
    boxes = results[0].boxes
    print(f"{img}: {len(boxes)} detections, classes: {[int(b.cls) for b in boxes]}, conf: {[round(float(b.conf),2) for b in boxes]}")
