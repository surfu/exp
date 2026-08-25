import cv2, numpy as np

cap = cv2.VideoCapture(0)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame,1)
    h,w,c = frame.shape

    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_img[:,:,1] = np.clip(hsv_img[:,:,1]*2, 0,255) 
    hsv_img[:,:,2] = np.clip(hsv_img[:,:,2]*5, 0,255) 

    img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)
    img = cv2.blur(img, (20,20))
    output_frame = cv2.addWeighted(frame,0.8,img,0.2,0)

    cv2.imshow('test', output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()