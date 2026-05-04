import requests
import time

MAC_IP = "10.132.229.14"
url = f"http://{MAC_IP}:6000/predict"

times = []
for i in range(21):
    start = time.time()
    with open('/home/pi/fire_project/fire-test.jpg', 'rb') as f:
        requests.post(url, files={'image': f})
    total = (time.time() - start) * 1000
    times.append(total)
    print(f"Run {i+1}: {total:.1f}ms")

avg = sum(times) / len(times)
print(f"\nAverage end-to-end cloud latency: {avg:.1f}ms")
