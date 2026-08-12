import cv2 as cv, numpy as np,mediapipe as mp, time, math

base, vis = mp.tasks.BaseOptions, mp.tasks.vision
options = vis.HandLandmarkerOptions(
    base_options=base(model_asset_path='hand_landmarker.task'),
    running_mode=vis.RunningMode.VIDEO, num_hands=2
)

mp_draw = vis.drawing_utils
mp_hands = vis.HandLandmarksConnections
mp_styles = vis.drawing_styles

cap = cv.VideoCapture(0)  # або шлях до відео 'video.mp4'
hue = 0
a = 100
with vis.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame,1)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        res = landmarker.detect_for_video(mp_img, int(time.time() * 1000))
        
        if res.hand_landmarks:
            for i in range(len(res.hand_landmarks)):
                hand = res.hand_landmarks[i]
                mp_draw.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )
                dis = math.dist((hand[4].x, hand[4].y),(hand[8].x,hand[8].y))*100
            print(dis)
            if dis>=5 and dis <=45:
                a = np.interp(dis,[2,43], [5,170])
                
        # 1. Отримуємо маску (біле/чорне)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        _, binary = cv.threshold(gray, a, 255, cv.THRESH_BINARY)

        # 2. Змінюємо відтінок з кожним кадром (0-179)
        hue = (hue + 2) % 180

        # 3. Створюємо 3-канальну картинку (BGR) і заповнюємо кольором
        result = np.zeros_like(frame)

        # Задаємо динамічний колір у HSV, потім переводимо у BGR
        # Для ЧОРНИХ зон (фон)
        bg_hsv = np.uint8([[[hue, 255, 255]]])
        bg_bgr = cv.cvtColor(bg_hsv, cv.COLOR_HSV2BGR)[0][0]

        # Для БІЛИХ зон (силует) — протилежний колір (+90 за спектром)
        fg_hsv = np.uint8([[[(hue + 90) % 180, 255, 255]]])
        fg_bgr = cv.cvtColor(fg_hsv, cv.COLOR_HSV2BGR)[0][0]

        # 4. Фарбуємо маску
        result[binary == 0] = bg_bgr  # Замінюємо чорне
        result[binary == 255] = fg_bgr  # Замінюємо біле

        cv.imshow("test", result)
        if cv.waitKey(1) & 0xFF == ord('q') or cv.getWindowProperty("test", cv.WND_PROP_VISIBLE) < 1: break

cap.release()
cv.destroyAllWindows()