from flask import Flask, request, jsonify
from ultralytics import YOLO
import time

app = Flask(__name__)
model = YOLO('/Users/akshaymp/Downloads/Fire model/fire_nano_trained.pt')

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    image_file.save('received.jpg')
    start = time.time()
    results = model('received.jpg')
    inference_ms = (time.time() - start) * 1000
    detections = len(results[0].boxes)
    print(f"Inference: {inference_ms:.1f}ms | Detections: {detections}")
    return jsonify({'status': 'done'})

app.run(host='0.0.0.0', port=6000)