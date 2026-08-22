import cv2, numpy as np

cap = cv2.VideoCapture(0)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame,1)
    h,w,c = frame.shape  
    frame = frame.astype(np.float32)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _,gray = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)

    noise = np.random.normal(0,1000,(h,w,c)).astype(np.float32)
    noise= np.clip(noise + 128, 0, 255).astype(np.uint8)

    stratched = np.tile(noise[:,w//2,:][:, np.newaxis, :], (1,w,1))
    stratched = cv2.applyColorMap(stratched,cv2.COLORMAP_INFERNO)
    stratched = stratched.astype(np.float32)

    # output_frame = cv2.addWeighted(stratched,0.5,frame,0.5,0)
    # output_frame = cv2.convertScaleAbs(output_frame, alpha=1.2, beta=-20)
    output_frame = np.zeros_like(frame, dtype=np.uint8)
    output_frame[gray==0] = stratched[gray==0] 
    output_frame[gray==255] = frame[gray==255].astype(np.uint8)

    cv2.imshow('test', output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()