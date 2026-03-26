import cv2
import json
import os

video_path = 'data/video3.mp4'

cap = cv2.VideoCapture(video_path)
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS):.1f}")
print(f"Длина: {cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS):.0f}c")

cap.set(cv2.CAP_PROP_POS_MSEC, 5000)
ret, frame = cap.read()
cap.release()
roi = cv2.selectROI("Select table ROI", frame, showCrosshair=True, fromCenter=False)

cv2.destroyAllWindows()
print("ROI:", roi) 
video_name = os.path.basename(video_path)
config = {
    "video_name": video_name,
    "roi": [int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])],
    "table_id": 1
}

with open("config/table_roi.json", "w") as f:
    json.dump(config, f, indent=2)

print("Сохранено в config/table_roi.json")