import cv2

ban = cv2.imread('./banmen4.png')

H = 324
W = 132
h_start = 70

jiban = ban[h_start : h_start + H, 90 : 90 + W]
eban = ban[h_start : h_start + H, 415 : 415 + W]
next = ban[73 : 123 , 240 : 260]
next2 = ban[132 : 172 , 259 : 274]

cv2.imwrite('./net/jiban.png', jiban)
cv2.imwrite('./net/eban.png', eban)
cv2.imwrite('./net/next.png', next)
cv2.imwrite('./net/next2.png', next2)