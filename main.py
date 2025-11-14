import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import csv

VIDEO_PATH = "challenge-mle2 copy.mp4"
OUTPUT_VIDEO = "output_tracking.mp4"
OUTPUT_CSV = "tracks.csv"

MODEL_PATH = "yolov8n.pt"
CONF_THRESH = 0.25

#COCO class IDs for person(pedestrian) = 0 and car = 2
TARGET_CLASSES = [0,2]
CLASS_NAMES = {0: "Pedestrian", 2: "Car"}

#assumption of distance to pixel ratio for calculations
METERS_PER_PIXEL = 0.2
TEXT_COLOR = (255, 255, 255)
BOX_COLOR = (0, 200, 255)

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30.0

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))

csv_file = open(OUTPUT_CSV, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame_id", "object_id", "class", "x1", "y1", "x2", "y2", "velocity_m_s"])

track_history = defaultdict(lambda: deque(maxlen=8))
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    results = model.track(
        frame,
        persist=True,
        conf=CONF_THRESH,
        classes=TARGET_CLASSES,
        tracker="bytetrack.yaml"
    )

    if results[0].boxes is None:
        out.write(frame)
        continue

    boxes = results[0].boxes.xyxy.cpu().numpy()
    ids = results[0].boxes.id
    cls = results[0].boxes.cls.cpu().numpy()
    if ids is None:
        out.write(frame)
        continue

    ids = ids.int().cpu().tolist()
    annotated = frame.copy()

    for box, obj_id, cls_id in zip(boxes, ids, cls):
        x1, y1, x2, y2 = map(int, box)
        label = CLASS_NAMES.get(int(cls_id), f"cls{int(cls_id)}")

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        track_history[obj_id].append((cx, cy, frame_id))

        #velocity calculation
        vel_m_s = 0.0
        hist = track_history[obj_id]
        if len(hist) >= 2:
            (x_prev, y_prev, f_prev) = hist[0]
            (x_now, y_now, f_now) = hist[-1]
            dpx = np.hypot(x_now - x_prev, y_now - y_prev)
            dframes = max(1, f_now - f_prev)
            px_per_s = (dpx * fps) / dframes
            vel_m_s = px_per_s * METERS_PER_PIXEL

        
        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, 2)

        label_text = f"{label} ID:{obj_id}"
        speed_text = f"{vel_m_s:.1f} m/s"
        font_scale = 0.5
        thickness = 2

        (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        (speed_w, speed_h), _ = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        pad = 2
        bg_y1 = y2 + 5
        cv2.rectangle(annotated, 
                      (x1, bg_y1),
                      (x1 + max(label_w, speed_w) + pad * 2, 
                       bg_y1 + label_h + speed_h + pad * 4),
                      (0, 0, 0), 
                      -1)
        
        cv2.putText(annotated, label_text, (x1 + pad, bg_y1 + label_h + pad),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, TEXT_COLOR, thickness)
        cv2.putText(annotated, speed_text, (x1 + pad, bg_y1 + label_h + speed_h + pad * 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
        
        csv_writer.writerow([frame_id, obj_id, label, x1, y1, x2, y2, f"{vel_m_s:.3f}"])

    out.write(annotated)
    
    cv2.imshow("MOT Tracking", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()   