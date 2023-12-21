import cv2

def onMouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)

img = cv2.imread('enemy.png')
cv2.imshow('sample', img)
cv2.setMouseCallback('sample', onMouse)
cv2.waitKey(0)