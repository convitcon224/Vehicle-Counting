import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from yolov9.models.common import DetectMultiBackend, AutoShape

# Config value
video_path = "data_ext/asia_-_31785 (540p) (online-video-cutter.com).mp4"
conf_threshold = 0.47
tracking_class_car = 2 # None: track all
tracking_class_motorbike = 3

tracking_classes = [2,3,5]

# Khởi tạo DeepSort
tracker = DeepSort(max_age=27)

# Khởi tạo YOLOv9
model  = DetectMultiBackend(weights="yolov9/yolov9-e-converted.pt",  fuse=True )
model  = AutoShape(model)

# Load classname từ file classes.names
with open("data_ext/classes.names") as f:
    class_names = f.read().strip().split('\n')

colors = np.random.randint(0,255, size=(len(class_names),3 ))
tracks = []

# Khởi tạo VideoCapture để đọc từ file video
cap = cv2.VideoCapture(video_path)

number_moto = 0
number_car = 0

# Tiến hành đọc từng frame từ video
while True:
    # Đọc
    ret, frame = cap.read()
    if frame is None:
        break
    if not ret:
        continue
    # Đưa qua model để detect
    results = model(frame)

    detect = []
    for detect_object in results.pred[0]:
        label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)

        if class_id not in tracking_classes or confidence < conf_threshold:
            continue

        detect.append([ [x1, y1, x2-x1, y2 - y1], confidence, class_id ])


    # Cập nhật,gán ID bằng DeepSort
    tracks = tracker.update_tracks(detect, frame = frame)

    # Vẽ lên màn hình các khung chữ nhật kèm ID
    for track in tracks:
        if track.is_confirmed():
            # Đếm xe
            if track.age - track._max_age == 7:
                if track.get_det_class() == 2:
                    number_car += 1
                if track.get_det_class() == 3:
                    number_moto += 1
                    
            # Lấy toạ độ, class_id để vẽ lên hình ảnh
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            color = colors[class_id]
            B, G, R = map(int,color)

            label = "{}-{}".format(class_names[class_id], track.track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show hình ảnh lên màn hình
    cv2.imshow("OT", frame)
    # Bấm Q thì thoát
    if cv2.waitKey(1) == ord("q"):
        break

print(f"Number of car: {number_car}")
print(f"Number of motobike: {number_moto}")
cap.release()
cv2.destroyAllWindows()