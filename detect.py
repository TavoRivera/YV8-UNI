import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(1)
cap.set(3,1280)
cap.set(4,720)

model = YOLO('models/best.pt')

while True:
    ret, frame = cap.read()
    t = cv2.waitKey(5)
    
    results = model.predict(frame, imgsz = 640, conf = 0.6)
    
    if len(results) !=0:
        for res in results:
          print('Perfums Detect')    
        
        annotated_frames = results[0].plot()
        
    cv2.imshow('Perfums Detect', annotated_frames) 
    
    t = cv2.waitKey(5)
    
    if t==27:
        break
    
cap.release()
cv2.destroyAllWindows()
