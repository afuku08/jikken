import sys,cv2

# 対象画像読み込み
img = cv2.imread("./tmp/puyo35.png",cv2.IMREAD_COLOR)

# 画像の大きさを取得
height, width, channels = img.shape[:3]
print("width: " + str(width))
print("height: " + str(height))
