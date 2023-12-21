import cv2
import numpy as np

img = cv2.imread("banmen3.png")

H = 324
W = 132
h_start = 70
player1_field = img[h_start : h_start + H, 90 : 90 + W]
player2_field = img[h_start : h_start + H, 415 : 415 + W]
fields = []
count = 1
for field in [player1_field, player2_field]:
    init_field = np.zeros((12, 6), dtype=np.uint8)

    h_unit = H // 12
    w_unit = W // 6    
        
    for h in range(0, H, h_unit):
        for w in range(0, W, w_unit):
            grid = field[h : h + h_unit, w : w + w_unit]
            cv2.imwrite("./tmp/puyo{0}.png" .format(count), grid)
            count += 1