import cv2 as cv, numpy as np

cap = cv.VideoCapture(0)
lego_tile = cv.imread("lego_tile.png", cv.IMREAD_UNCHANGED)
window_name = "Lego Mosaic w Texture"
cv.imshow(window_name, np.zeros((480, 500, 3), dtype=np.uint8))
cv.createTrackbar("Brick Size", window_name, 40, 100,lambda x: None )
while True:
    run, img = cap.read()
    if not run:
        break
    img = cv.flip(img,1)
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img[:,:,1] = np.clip(img[:,:,1]*1.5,0,255)
    img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
    brick_size = cv.getTrackbarPos("Brick Size", window_name)
    if brick_size<2:
        brick_size = 2
    h, w = img.shape[:2]

    w = (w // brick_size) * brick_size
    h = (h // brick_size) * brick_size
    img = cv.resize(img, (w, h))

    cols = w // brick_size
    rows = h // brick_size
    small = cv.resize(img, (cols, rows), interpolation=cv.INTER_LINEAR)
    result = cv.resize(small, (w, h), interpolation=cv.INTER_NEAREST)

    lego_resized = cv.resize(lego_tile, (brick_size, brick_size))

    if lego_resized.shape[2] == 4:
        b_tile, g_tile, r_tile, alpha = cv.split(lego_resized)
        tile_rgb = cv.merge([b_tile, g_tile, r_tile]).astype(float)
        alpha_f = alpha.astype(float) / 255.0
        alpha_3c = cv.merge([alpha_f, alpha_f, alpha_f])
    else:
        tile_rgb = lego_resized[:, :, :3].astype(float)
        alpha_3c = np.ones((brick_size, brick_size, 3), dtype=float)

    for y in range(0, h, brick_size):
        for x in range(0, w, brick_size):
            roi = result[y : y + brick_size, x : x + brick_size].astype(float)
            blended = tile_rgb * alpha_3c + roi * (1.0 - alpha_3c)
            result[y : y + brick_size, x : x + brick_size] = blended.astype(np.uint8)

    cv.imshow(window_name, result)
    if cv.waitKey(1) & 0xFF == ord('q') or cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1: break
cap.release()
cv.destroyAllWindows()