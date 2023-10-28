import cv2

img = cv2.imread('banmen.png')
#cv2.imshow('OpenCv', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#height, width, channels = img.shape[:3]
#print("width: " + str(width))
#print("height: " + str(height))

def mosaic(src, ratio=0.1):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

def small(src, ratio=0.1):
    return cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)


cv2.imshow('banmen', img)
cv2.waitKey(0)
#dst_01 = mosaic(jibanmen,0.05)
#cv2.imshow('jiban', dst_01)
#cv2.waitKey(0)
cv2.destroyAllWindows()

jibanmen = img[71:390, 90:222 ]
cv2.imshow('jibanmen', jibanmen)
cv2.waitKey(0)
cv2.destroyAllWindows()

puyo_x = 22
puyo_y = 26
for i in range(12):
    puyo1 = jibanmen[i*puyo_y:(i+1)*puyo_y, 0:puyo_x]
    puyo2 = jibanmen[i*puyo_y:(i+1)*puyo_y, puyo_x:puyo_x*2]
    puyo3 = jibanmen[i*puyo_y:(i+1)*puyo_y, puyo_x*2:puyo_x*3]
    puyo4 = jibanmen[i*puyo_y:(i+1)*puyo_y, puyo_x*3:puyo_x*4]
    puyo5 = jibanmen[i*puyo_y:(i+1)*puyo_y, puyo_x*4:puyo_x*5]
    puyo6 = jibanmen[i*puyo_y:(i+1)*puyo_y, puyo_x*5:puyo_x*6]
    '''puyo1 = jibanmen[0:55, 1:67]
    puyo1 = jibanmen[0:55, 68:134]
    puyo1 = jibanmen[0:55, 135:201]
    puyo1 = jibanmen[0:55, 202:268]
    puyo1 = jibanmen[0:55, 269:335]
    puyo1 = jibanmen[0:55, 336:400]
    small_puyo1 = small(puyo1,0.01)
    small_puyo2 = small(puyo2)
    small_puyo3 = small(puyo3)
    small_puyo4 = small(puyo4)
    small_puyo5 = small(puyo5)
    small_puyo6 = small(puyo6)'''
    
    cv2.imshow('puyo1', puyo1)
    cv2.imshow('puyo2', puyo2)
    cv2.imshow('puyo3', puyo3)
    cv2.imshow('puyo4', puyo4)
    cv2.imshow('puyo5', puyo5)
    cv2.imshow('puyo6', puyo6)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    