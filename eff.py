import cv2
import numpy as np

cap = cv2.VideoCapture(0) # або 0 для веб-камери

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Крок 1. Роблимо «абстракцію»: сильно зменшуємо кадр (наприклад, до 16x16 пікселів)
    # Пікселі змішуються у суцільні кольорові плями (як розмазана хмара)
    small_size = 20
    temp = cv2.resize(frame, (small_size, small_size), interpolation=cv2.INTER_LINEAR)
    
    # Крок 2. Розтягуємо назад до оригінального розміру з кутовою/лінійною інтерполяцією
    # Це створює ефект розмитих плазмових хмар або розфокусованого світла
    abstract_cloud = cv2.resize(temp, (w, h), interpolation=cv2.INTER_CUBIC)

    # Крок 3. Додаємо ефект затемнення / містичного туману (Vignette / Multiply)
    # Створюємо маску затемнення по краях або по всій площині
    kernel_x = cv2.getGaussianKernel(w, w/2)
    kernel_y = cv2.getGaussianKernel(h, h/2)
    mask = kernel_y * kernel_x.T
    mask = mask / mask.max() # нормалізуємо від 0 до 1
    
    # Застосовуємо затемнення до нашої хмари
    dark_cloud = np.uint8(abstract_cloud * mask[:, :, np.newaxis])

    # Опціонально: можна змішати оригінал з абстрактною хмарою (накладання)
    # blend = cv2.addWeighted(frame, 0.2, dark_cloud, 0.8, 0)

    cv2.imshow('Abstract Cloud Effect', dark_cloud)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()