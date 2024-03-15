import cv2
from ultralytics import YOLO
import math
from sort import *

save_name = "output.mp4"
fps = 10
width = 600
height = 480
output_size = (width, height)
out = cv2.VideoWriter(save_name,cv2.VideoWriter_fourcc('M','J','P','G'), fps , output_size )


cap = cv2.VideoCapture('car.mp4')
model = YOLO('yolov8s.pt')

classnames = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
              'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
              'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 
              'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
              'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'potted plant', 'bed', 'dining table',
              'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
              'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

tracker = Sort(max_age=25, min_hits=3, iou_threshold=0.3)

lst = []
# limits = [282, 308, 1004, 308]
limits = [50, 200, 300, 200]
total_count = 0

#line
text_color = (255,255,255)
red_color = (0,0,255)


while True:
    success, img = cap.read()
    
    results = model(img, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            current_class = classnames[cls]

            if conf > 0.3 and current_class == 'car':
                
                # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.putText(img, f'{classnames[cls]} {conf}', (x1,y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255,0,0), 2)

                current_array = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_array))

    results_tracker = tracker.update(detections)

    # Line
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), red_color, 2)

    for result in results_tracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 1)
        cv2.putText(img, f'{int(id)}', (x1, y1), 1, 0.7, (255,0,0), 1)

        cx = int(x1+x2)//2
        cy = int(y1+y2)//2
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

        if limits[0]< cx < limits[2] and limits[1] - 5 < cy < limits[1] + 5:
            if id not in lst:
                lst.append(id)
                total_count += 1
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0,255,0), 2)
    
    cv2.putText(img, f'Count: {total_count}', (50, 50), 2, 2, (255, 89,0), 3)
    
    cv2.imshow("Image", img)
    out.write(cv2.resize(img, output_size ))

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()