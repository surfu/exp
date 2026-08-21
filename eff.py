import cv2
import numpy as np

def nothing(x):
    pass

window_name = "Control Panel"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar("Mode (True/False)", window_name, 0,1, nothing)

cap = cv2.VideoCapture(0)

accumulated_space = None

while cap.isOpened():
    is_active = bool(cv2.getTrackbarPos("Mode (True/False)", window_name))
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)

    if not is_active:
        output_frame = frame
    else:
        h, w = frame.shape[:2]
        
        dust_noise = np.random.normal(0, 25, (h, w, 3)).astype(np.float32)

        nebula = cv2.GaussianBlur(frame, (31, 31), 0)
        nebula = np.float32(nebula) + dust_noise
        
        if accumulated_space is None:
            accumulated_space = nebula
        else:
            accumulated_space = cv2.addWeighted(accumulated_space, 0.85, nebula, 0.15, 0)

        final_space = np.clip(accumulated_space, 0, 255).astype(np.uint8)

        gray_nebula = cv2.cvtColor(final_space, cv2.COLOR_BGR2GRAY)
        twilight_glow = cv2.applyColorMap(gray_nebula, cv2.COLORMAP_TWILIGHT)
        
        twilight_glow = cv2.cvtColor(twilight_glow, cv2.COLOR_BGR2HSV)
        twilight_glow[:, :, 2] = 255 - twilight_glow[:, :, 2] 
        twilight_glow = cv2.cvtColor(twilight_glow, cv2.COLOR_HSV2BGR)

        twilight_glow = cv2.GaussianBlur(twilight_glow, (55, 55), 0)

        final_space = cv2.addWeighted(final_space, 0.2, twilight_glow, 0.8, 0)
        output_frame = cv2.convertScaleAbs(final_space, alpha=1.2, beta=-20)

    cv2.imshow(window_name, output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()