import cv2
import numpy as np

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
while cap.isOpened():
    run, img = cap.read()
    if not run:
        break
    
    h, w = img.shape[:2]
    cell_size = 10
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    halftone_output = np.zeros((h, w, 3), dtype=np.uint8)
    
    for y in range(0, h, cell_size):
        for x in range(0, w, cell_size):
            cell = gray[y : y + cell_size, x : x + cell_size]
            if cell.size == 0:
                continue
            avg_val = np.mean(cell)
            
            radius = int(cell_size * 0.5 * (avg_val / 255.0))
            center_x = x + cell_size // 2
            center_y = y + cell_size // 2
            
            if radius > 0:
                cv2.circle(halftone_output, (center_x, center_y), radius, (255, 255, 255), -1)

    _, gray_thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    blur = cv2.blur(gray_thresh, (15, 15))
    color_mapped_blur = cv2.applyColorMap(blur, cv2.COLORMAP_OCEAN)

    b, g, r = cv2.split(color_mapped_blur)
    rgba_img = cv2.merge([b, g, r, gray_thresh])

    color_mapped_img = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
    img_bgra = cv2.cvtColor(color_mapped_img, cv2.COLOR_BGR2BGRA)

    base_blend = cv2.addWeighted(img_bgra, 0.2, rgba_img, 0.8, 0)

    halftone_bgra = cv2.cvtColor(halftone_output, cv2.COLOR_BGR2BGRA)
    output = cv2.addWeighted(base_blend, 0.4, halftone_bgra, 0.6, 0)

    cv2.imshow('Title', output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()