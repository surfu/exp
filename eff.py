import cv2
import numpy as np

width, height = 1000, 778
canva = np.zeros((height, width, 3), dtype=np.uint8)

phase_x1 = np.random.uniform(0, 50, 3)
phase_y1 = np.random.uniform(0, 50, 3)
phase_x2 = np.random.uniform(0, 50, 3)
phase_y2 = np.random.uniform(0, 50, 3)
t = 0.0

while True: 
    canva = (canva.astype(np.float32) * 0.92).astype(np.uint8)

    center = (width / 2, height / 2)
    M = cv2.getRotationMatrix2D(center, angle=0.2, scale=1.005)
    canva = cv2.warpAffine(canva, M, (width, height), flags=cv2.INTER_LINEAR)

    cx1 = int(width / 2 + np.sin(t * 1.1 + phase_x1[0]) * 180 + np.sin(t * 2.7 + phase_x1[1]) * 90)
    cy1 = int(height / 2 + np.cos(t * 0.9 + phase_y1[0]) * 140 + np.sin(t * 2.4 + phase_y1[1]) * 70)
    cx2 = int(width / 2 + np.cos(t * 1.4 + phase_x2[0]) * 160 + np.sin(t * 2.2 + phase_x2[1]) * 100)
    cy2 = int(height / 2 + np.sin(t * 1.3 + phase_y2[0]) * 150 + np.cos(t * 2.9 + phase_y2[1]) * 80)

    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(overlay, (cx1, cy1), 35, (255, 255, 255), -1)
    cv2.circle(overlay, (cx2, cy2), 20, (255, 255, 255), -1)

    overlay = cv2.GaussianBlur(overlay, (51, 51), 0)
    
    canva = cv2.add(canva, overlay)
    
    gray = cv2.cvtColor(canva, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray,(20,20))
    organic_colored = cv2.applyColorMap(gray, cv2.COLORMAP_PINK)

    cv2.imshow("Title", organic_colored)
    t += 0.01
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()