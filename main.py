import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from pillow_heif import register_heif_opener
import time

register_heif_opener()

def finder(image, kernel, thr, er, iter, channel, method):
    image = cv2.blur(image, (kernel, kernel))
    print(f'Average pixel {np.average(image[:,:,channel])}')
    print(f'Mean pixel {np.mean(image[:,:,channel])}')
    print(f'Max pixel {np.max(image[:,:,channel])}')
    _, thresh = cv2.threshold(image[:,:,channel], thr, 255, method)
    thresh = cv2.erode(thresh, np.ones((er, er), np.uint8), iterations=iter)
    contour, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnt = sorted(contour, key=cv2.contourArea)[-1]

    return thresh, cnt, contour

def main():
    pass
# 4522, 4537, 4539
file_number = '4522'
img = np.asanyarray(Image.open(f'images/heic/IMG_{file_number}.heic'))
assert img is not None, 'file could not be read'

h, w = img.shape[:2]
k = h / w
s = int(h / 8)    
image = cv2.resize(img.copy(), (int(s / k), s), interpolation=cv2.INTER_CUBIC)

#color_hist(image)
#plt.show()

# parameters of finder function:
# source image, kernel size, thresh value, erosian value, iterations, channel number, method
start = time.time()
thresh, cnt, c = finder(image, 5, 121.5, 1, 3, 1, cv2.THRESH_BINARY_INV)
mask = np.zeros(image.shape[:2], np.uint8)

M = cv2.drawContours(mask.copy(), [cnt], -1, (255,255,255), -1)

masked = cv2.subtract(cv2.bitwise_and(image, image, mask = M), 35)

print(np.max(masked))
thresh_1, cnt_1, cnts = finder(masked, 3, 0.8, 3, 3, 1, cv2.THRESH_BINARY)

cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
cv2.drawContours(image, [cnt_1], -1, (0, 0, 255), 2)

c = cv2.contourArea(cnt)
c_1 = cv2.contourArea(cnt_1)
bark = np.round(((c - c_1) / c) * 100, 2)

end = time.time()

print(f'Process time is: {end - start} s.')
print(f'Area of contour 1 = {c}')
print(f'Area of contour 2 = {c_1}')
print(f'Bark in % = {bark}%')

cv2.putText(image, f'Bark area = {bark}%', (20, image.shape[:2][0] - 40), cv2.FONT_HERSHEY_SIMPLEX, (s / 600), (255, 0, 0), int(round(s / 300)))

cv2.imshow('Original', image[:,:,::-1])

#cv2.imwrite(f'images/experiment/{file_number}.jpg', image)

cv2.waitKey()
cv2.destroyAllWindows()