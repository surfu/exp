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

run,clr = False ,0

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
lst = []
with vis.HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        print(fps)
        ret, frame = cap.read()
        if not ret: break

        # 1. BLUR CAPTURE FIRST (Strong blur on background only)
        img = cv.flip(frame, 1)

        # 2. crop 9:16
        h, w, _ = img.shape
        # img = cv.resize(img,(w // 2, h // 2))
        h,w,_= img.shape

        # 3. MediaPipe Tracking
        cv.rectangle(img, (120, 50),( 190, 120), (255,0,0), -1)
        cv.rectangle(img, (190, 50),( 260, 120), (0,255,0), -1)
        cv.rectangle(img, (260, 50),( 330, 120), (0,0,255), -1)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(img, cv.COLOR_BGR2RGB))
        res = landmarker.detect_for_video(mp_img, int(time.time() * 1000))

        if res.hand_landmarks:
            for i in range(len(res.hand_landmarks)):
                hand = res.hand_landmarks[i]
                mp_draw.draw_landmarks(
                    img, hand, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )
                hand_label = res.handedness[i][0].category_name
                is_sigh_r = (hand_label == "Right"and float(hand[5].x) > float(hand[4].x) and float(hand[5].y)<float(hand[4].y))
                is_sigh_l = (hand_label == "Left" and float(hand[5].x) < float(hand[4].x) and float(hand[5].y)>float(hand[4].y))
                is_sigh = False
                if is_sigh_l:
                    lst.append("a")
                    print('l')
                if is_sigh_r:
                    print('r')
                    lst.append("s")
                if len(lst) > 15:
                    lst = lst[-15:]
                if len(lst) >= 3 and lst[-3:] == ["a", "s", "a"]:
                    is_sigh = True
                if is_sigh_l:
                    print("ueu")
                    h, w = img.shape[:2]
                    film = img.astype(np.float32)
                    film[:, :, 0] *= 0.80
                    film[:, :, 1] *= 0.95
                    film[:, :, 2] *= 1.15
                    film = np.clip(film, 0, 255).astype(np.uint8)
                    grain = np.random.randint(-18, 18, (h, w, 3), dtype=np.int16)
                    film = np.clip(film.astype(np.int16) + grain, 0, 255).astype(np.uint8)
                    kernel_x = cv.getGaussianKernel(w, w / 2)
                    kernel_y = cv.getGaussianKernel(h, h / 2)
                    mask = (kernel_y * kernel_x.T) / (kernel_y * kernel_x.T).max()
                    for i in range(3):
                        film[:, :, i] = (film[:, :, i] * mask).astype(np.uint8)
                    img = film
                    img[:, :, 0] = img[:, :, 0] * 0.7  # Lower Blue
                    img[:, :, 2] = cv.add(img[:, :, 2], 30)  # Boost Red

        output.write(img)
        cv.imshow("test", img)
        if cv.waitKey(1) & 0xFF == ord('q') or cv.getWindowProperty("test", cv.WND_PROP_VISIBLE) < 1: break

cap.release()
cv.destroyAllWindows()