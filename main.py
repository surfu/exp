import cv2, numpy as np

def apply_vhs_effect(frame):
    
    return vhs_frame

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame,1)

    h, w, c = frame.shape

    shift = 3
    b, g, r = cv2.split(frame)
    
    r_shifted = np.roll(r, -shift, axis=1)
    b_shifted = np.roll(b, shift, axis=1)
    
    vhs_frame = cv2.merge([b_shifted, g, r_shifted])

    num_lines = np.random.randint(2, 6)
    for _ in range(num_lines):
        y = np.random.randint(0, h - 5)
        line_height = np.random.randint(1, 4)
        line_shift = np.random.randint(-15, 15)
        vhs_frame[y:y+line_height, :] = np.roll(vhs_frame[y:y+line_height, :], line_shift, axis=1)

    noise = np.random.randint(-15, 15, (h, w, c), dtype='int16')
    vhs_frame = np.clip(vhs_frame.astype('int16') + noise, 0, 255).astype('uint8')

    vhs_frame = cv2.GaussianBlur(vhs_frame, (3, 3), 0)

    cv2.putText(vhs_frame, "PLAY  0:00:12", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vhs_frame, "SEP 11 2026", (30, h - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("VHS Effect", vhs_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()