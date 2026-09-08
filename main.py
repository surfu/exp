import cv2
import numpy as np

def apply_green_dot_matrix(image_path, grid_step=6, max_radius=2.5):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Image not found")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    green_color = (0, 255, 65)

    for y in range(0, h, grid_step):
        for x in range(0, w, grid_step):
            intensity = gray[y, x]
        
            if intensity < 90:
                continue
                
            norm_intensity = intensity / 255.0

            radius = int(norm_intensity * max_radius)
            if radius > 0:
                cv2.circle(canvas, (x, y), radius, green_color, -1)
            else:
                canvas[y, x] = green_color

    blur = cv2.GaussianBlur(canvas, (5, 5), 0)
    result = cv2.addWeighted(canvas, 1.0, blur, 0.4, 0)

    return result

output_image = apply_green_dot_matrix("input.jpg", grid_step=5, max_radius=1.5)

cv2.imshow("Green Dot Effect", output_image)
cv2.imwrite("output_green_dot.png", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()