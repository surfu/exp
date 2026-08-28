import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while cap.isOpened():
    run, img = cap.read()
    if not run:
        break
    
    h, w = img.shape[:2]
    cell_size = 6
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Чистий білий холст для фінального рендеру
    output = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    # 2. Робимо крапочки тільки для сірої/напівтонової зони, а чорне малюємо суцільним
    for y in range(0, h, cell_size):
        for x in range(0, w, cell_size):
            cell = gray[y : y + cell_size, x : x + cell_size]
            if cell.size == 0:
                continue
            avg_val = np.mean(cell)
            
            # Якщо область темна (глибокий чорний) — заливаємо клітинку суцільним чорним без крапок
            if avg_val < 90:
                output[y : y + cell_size, x : x + cell_size] = (0, 0, 0)
            # Якщо занадто світла — залишаємо білою
            elif avg_val > 210:
                continue
            # Для сірого діапазону — малюємо крапочки
            else:
                radius = int(cell_size * 0.48 * (1.0 - (avg_val / 255.0)))
                center_x = x + cell_size // 2
                center_y = y + cell_size // 2
                if radius > 0:
                    cv2.circle(output, (center_x, center_y), radius, (0, 0, 0), -1)

    # 3. Твоя кольорова обробка Ocean для підсвічування контурів
    _, gray_thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    blur = cv2.blur(gray_thresh, (15, 15))
    color_mapped_blur = cv2.applyColorMap(blur, cv2.COLORMAP_OCEAN)

    b, g, r = cv2.split(color_mapped_blur)
    rgba_img = cv2.merge([b, g, r, gray_thresh])

    color_mapped_img = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
    img_bgra = cv2.cvtColor(color_mapped_img, cv2.COLOR_BGR2BGRA)

    base_blend = cv2.addWeighted(img_bgra, 0.2, rgba_img, 0.8, 0)

    # 4. Змішуємо твій неоновий стиль з нашим чітким чорним і крапковим сірим
    output_bgra = cv2.cvtColor(output, cv2.COLOR_BGR2BGRA)
    final_output = cv2.addWeighted(base_blend, 0.3, output_bgra, 0.7, 0)

    cv2.imshow('Title', final_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()