import cv2 as cv

# 1. Просто читаємо картинку
frame = cv.imread("pic.png")
h,w,_ = frame.shape
frame = cv.resize(frame,(w // 2, h // 2))

# 2. Обробка
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
_, binary = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)

result = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
result[binary == 0] = (0, 0, 100)  # Замінюємо чорний на темно-червоний

# 3. Показуємо і чекаємо
cv.imshow("PNG Frame", result)
cv.waitKey(0)  # Чекає натискання будь-якої клавіші для закриття
cv.destroyAllWindows()