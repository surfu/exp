import cv2 as cv, mediapipe as mp, random, time, math, numpy as np

base, vis = mp.tasks.BaseOptions, mp.tasks.vision
options = vis.HandLandmarkerOptions(
    base_options=base(model_asset_path='hand_landmarker.task'),
    running_mode=vis.RunningMode.VIDEO, num_hands=2
)

mp_draw = vis.drawing_utils
mp_hands = vis.HandLandmarksConnections
mp_styles = vis.drawing_styles

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

clol = list((
    "COLORMAP_AUTUMN",
    "COLORMAP_BONE",
    "COLORMAP_CIVIDIS",
    "COLORMAP_COOL",
    "COLORMAP_DEEPGREEN",
    "COLORMAP_HOT",
    "COLORMAP_HSV",
    "COLORMAP_INFERNO",
    "COLORMAP_JET",
    "COLORMAP_MAGMA",
    "COLORMAP_OCEAN"
))
hue =0
with vis.HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv.flip(frame,1)
        # 1. Створюємо маску (біле/чорне)
    # Перетворюємо в сірий
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Створюємо бінарну маску. Наприклад, поріг 100.
        # Тіні стануть чорними (0), світлі зони — білими (255).
        _, binary = cv.threshold(gray, 40, 255, cv.THRESH_BINARY)

        # Іноді краще інвертувати маску, якщо об'єкт темніший за фон.
        # binary = cv.bitwise_not(binary)

        # 2. Генеруємо динамічний колір для ТЛА (фон)
        hue = (hue + 2) % 180  # Плавний перелив кольорів

        # Отримуємо BGR колір з одного HSV пікселя
        bg_color_hsv = np.uint8([[[hue, 255, 255]]])
        bg_color_bgr = cv.cvtColor(bg_color_hsv, cv.COLOR_HSV2BGR)[0][0]

        # Створюємо суцільний кольоровий кадр такого ж розміру, як оригінал
        # Перетворюємо кортеж bg_color_bgr (R,G,B) у потрібний тип даних для NumPy
        color_background = np.full(frame.shape, bg_color_bgr, dtype=np.uint8)

        # 3. КОМБІНУЄМО ЗОБРАЖЕННЯ

        # Крок А: Вирізаємо СИЛУЕТ з реального зображення (біла зона маски)
        # mask=binary каже функції брати пікселі з frame тільки там, де маска біла.
        foreground_part = cv.bitwise_and(frame, frame, mask=binary)

        # Крок Б: Вирізаємо ФОН з кольорового кадру (чорна зона маски)
        # Спочатку інвертуємо маску, щоб чорне стало білим.
        mask_inv = cv.bitwise_not(binary)
        background_part = cv.bitwise_and(color_background, color_background, mask=mask_inv)

        # Крок В: Додаємо обидві частини разом
        # Оскільки зони не перекриваються (одна чорна там, де інша кольорова), add працює ідеально.
        result = cv.add(foreground_part, background_part)

        cv.imshow("test", result)
        if cv.waitKey(1) & 0xFF == ord('q') or cv.getWindowProperty("test", cv.WND_PROP_VISIBLE) < 1: break

cap.release()
cv.destroyAllWindows()