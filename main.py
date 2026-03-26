import cv2
import pandas as pd
import numpy as np
import json
import argparse
import os
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Детекция уборки столиков")
parser.add_argument("--video", required=True, help="Путь к видео")
args = parser.parse_args()
video_name = os.path.splitext(os.path.basename(args.video))[0]  # "video1"
output_video = f"output/{video_name}_output.mp4"
output_events = f"output/{video_name}_events.csv"  
output_stats = f"output/{video_name}_stats.txt"

with open("config/table_roi.json") as f:
    cfg = json.load(f)

roi = cfg['roi']
x, y, w, h = roi

video_path = args.video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Всего кадров: {total_frames}")

model = YOLO("yolov8s.pt")

def is_table_occupied(frame, roi, model) -> bool:
    x, y, w, h = roi
    roi_frame = frame[y:y+h, x:x+w]
    if roi_frame.size == 0:
        return False
    
    results = model(roi_frame, classes=[0], conf=0.1, imgsz=640, verbose=False)

    if len(results) == 0:
        return False
    boxes = results[0].boxes
    return boxes is not None and len(boxes) > 0

def format_time_full(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

current_state = "EMPTY"
events = []
last_event_time = None
frame_idx = 0
EMPTY_COUNTER = 0
OCCUPIED_COUNTER = 0
FRAME_SKIP = 3 
MIN_FRAMES_CONFIRM = 15

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % FRAME_SKIP == 0:
        occupied = is_table_occupied(frame, roi, model)
    timestamp = frame_idx / fps

    if occupied:
        OCCUPIED_COUNTER += 1
        EMPTY_COUNTER = 0
    else:
        EMPTY_COUNTER += 1
        OCCUPIED_COUNTER = 0

    new_state = current_state
    if OCCUPIED_COUNTER >= MIN_FRAMES_CONFIRM:
        new_state = "OCCUPIED"
    elif EMPTY_COUNTER >= MIN_FRAMES_CONFIRM:
        new_state = "EMPTY"
    
    if new_state != current_state:
        event_type = "approach" if new_state == "OCCUPIED" else "leave"
        events.append({
            "frame": frame_idx,
            "time_sec": round(timestamp, 1),
            "event": event_type,
            "state": new_state
        })
        print(f"✅ СОБЫТИЕ: {event_type} на {timestamp:.1f}с")  # для дебага
        current_state = new_state

    color = (0, 255, 0) if current_state == "EMPTY" else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.putText(frame, current_state, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    out.write(frame)
    frame_idx += 1
    if frame_idx % 100 == 0:  # каждые 100 кадров
        progress = (frame_idx / total_frames) * 100
        print(f"\rПрогресс: {progress:.1f}% ({frame_idx}/{total_frames})", end="")


df = pd.DataFrame(events)
df.to_csv(output_events, index=False)

wait_times = []
for i in range(len(df) - 1):
    if df.iloc[i]["event"] == "leave" and df.iloc[i+1]["event"] == "approach":
        wait_time = df.iloc[i+1]['time_sec'] - df.iloc[i]['time_sec']  # одинаковые кавычки
        wait_times.append(wait_time)

if wait_times:
    mean_wait = np.mean(wait_times)
    print(f"✅ Среднее время ожидания: {mean_wait:.1f} сек ({mean_wait/60:.1f} мин)")
    with open(output_stats, "w") as f:
        f.write(f"Видео: {video_name}\nСреднее время: {mean_wait:.1f} сек\n")
else:
    print("⚠️ Пар 'уход→подход' не найдены")

cap.release()
out.release()