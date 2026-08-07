import cv2 as cv, mediapipe as mp, random, time

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

active_words, last_spawn_time, font_size, word, SPAWN_DELAY = [], 0, 1.2, "I Love U", 0.3
start_y = -(len(word) - 1) * 40

with vis.HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. BLUR CAPTURE FIRST (Strong blur on background only)
        blurred_frame = cv.blur(frame, (51, 51))
        img = cv.flip(frame, 1)

        # 2. Crop 9:16
        h, w, _ = img.shape
        crop = cv.resize(img,(w // 2, h // 2))
        ch, cw, _ = crop.shape

        # 3. MediaPipe Tracking
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(crop, cv.COLOR_BGR2RGB))
        res = landmarker.detect_for_video(mp_img, int(time.time() * 1000))

        if res.hand_landmarks:
            for hand in res.hand_landmarks:
                mp_draw.draw_landmarks(
                    crop, hand, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )
                is_peace_sign = (hand[12].y < hand[9].y and 
                                 hand[8].y < hand[5].y and 
                                 hand[16].y > hand[13].y)
                curr_time = time.time()
                if is_peace_sign and (curr_time - last_spawn_time > SPAWN_DELAY):
                    active_words.append({
                        'x': random.randint(1, w - 30),
                        'y': start_y,
                        'font_size': float(random.randint(9, 13)) / 10
                    })
                    last_spawn_time = curr_time
                if is_peace_sign:
                    crop = cv.blur(crop,(20,20))
                    for item in active_words[:]:
                        item['y'] += 10
                        
                        for j in range(len(word)):
                            char_y = item['y'] + j * 40
                            if char_y >= 0:
                                cv.putText(crop, word[j], (item['x'], char_y), 
                                        cv.FONT_HERSHEY_PLAIN, item['font_size'], (0, 0, 150), 2, cv.LINE_AA)

                        if item['y'] >= ch:
                            active_words.remove(item)

        cv.imshow("test", crop)
        if cv.waitKey(1) & 0xFF == ord('q') or cv.getWindowProperty("test", cv.WND_PROP_VISIBLE) < 1: break

cap.release()
cv.destroyAllWindows()