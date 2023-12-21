import os

import cv2
import numpy as np
import collections


puyo_cont = []

def get_field_info(img):
    H = 324
    W = 132
    h_start = 70
    player1_field = img[h_start : h_start + H, 90 : 90 + W]
    player2_field = img[h_start : h_start + H, 415 : 415 + W]
    fields = []
    for field in [player1_field, player2_field]:
        init_field = np.zeros((12, 6), dtype=np.uint8)

        h_unit = H // 12
        w_unit = W // 6    
        
        for h in range(0, H, h_unit):
            for w in range(0, W, w_unit):
                grid = field[h : h + h_unit, w : w + w_unit]
                #cv2.imshow('banmen', grid)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                puyo = classifier.predict(grid, template_type="field")
                print(puyo)
                #print(puyo)
                this_puyo = -1
                #からの場合
                if puyo == 6:
                    this_puyo = 0
                #お邪魔ぷよの場合
                elif puyo == 5:
                    this_puyo = 5
                #通常ぷよの場合
                else:
                    result = puyo not in puyo_cont
                    if len(puyo_cont) == 4:
                        if result:
                            this_puyo = 0
                    else:
                        if result:
                            puyo_cont.append(puyo)
                        
                    this_puyo = puyo_cont.index(puyo) + 1
                    
                init_field[h // h_unit, w // w_unit] = this_puyo
                
        init_field = field_edit(init_field)
        fields.append(init_field)
    return fields


def get_puyo_judge(img):
    judge_img = img[200:300, 0:100].reshape(-1)
    template_judge_img = cv2.imread("images/judge_img.jpg").reshape(-1)
    diff = np.mean(judge_img - template_judge_img)
    return judge_img


def get_next_puyo_info(img):
    player1_next = img[73 : 123 , 240 : 260]
    h, w, c = player1_next.shape
    player1_next = player1_next[: h // 2], player1_next[h // 2 :]
    player1_next = [classifier.predict(i, template_type="p1") for i in player1_next]

    player1_next_next = img[132 : 172 , 259 : 274]
    h, w, c = player1_next_next.shape
    player1_next_next = player1_next_next[: h // 2], player1_next_next[h // 2 :]
    player1_next_next = [
        classifier.predict(i, template_type="p1") for i in player1_next_next
    ]

    player2_next = img[73 : 123 , 377 : 397]
    h, w, c = player2_next.shape
    player2_next = player2_next[: h // 2], player2_next[h // 2 :]
    player2_next = [classifier.predict(i, template_type="p2") for i in player2_next]

    player2_next_next = img[132 : 172 , 364 : 379]
    h, w, c = player2_next_next.shape
    player2_next_next = player2_next_next[: h // 2], player2_next_next[h // 2 :]
    player2_next_next = [
        classifier.predict(i, template_type="p2") for i in player2_next_next
    ]

    output = [player1_next, player1_next_next, player2_next, player2_next_next]
    player1_nexts = [player1_next, player1_next_next]
    #player2_nexts = [player2_next, player2_next_next]
    #output = [player1_nexts, player2_nexts]
    return player1_nexts


class puyo_classifier(object):
    def __init__(self, puyo_types):
        self._puyo_types = puyo_types
        self._field_template = {}
        for name in self._puyo_types:
            img = cv2.resize(cv2.imread(f"images/field/{name}.jpg"), (20, 20))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            self._field_template[name] = img

        self._p1_template = {}
        for name in self._puyo_types[:-2]:
            img = cv2.resize(cv2.imread(f"images/p1/p1_{name}.jpg"), (20, 20))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            self._p1_template[name] = img

        self._p2_template = {}
        for name in self._puyo_types[:-2]:
            img = cv2.resize(cv2.imread(f"images/p2/p2_{name}.jpg"), (20, 20))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            self._p2_template[name] = img

    def predict(self, img, template_type="field"):
        img = cv2.resize(img, (20, 20))
        differences = []

        channel = 0

        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hist = cv2.calcHist([img], [channel], None, [256], [0, 256])

        for name in self._puyo_types[:-2]:
            if template_type == "field":
                template_img = self._field_template[name]
            elif template_type == "p1":
                template_img = self._p1_template[name]
            elif template_type == "p2":
                template_img = self._p2_template[name]
            template_img_hist = cv2.calcHist(
                [template_img], [channel], None, [256], [0, 256]
            )
            template_img_hist = cv2.calcHist(
                [template_img], [channel], None, [256], [0, 256]
            )
            template_img_hist = cv2.calcHist(
                [template_img], [channel], None, [256], [0, 256]
            )

            diff = cv2.compareHist(template_img_hist, img_hist, 0)
            differences.append(diff)

        if template_type == "field":
            channel = 0
            img_hist = cv2.calcHist([img], [channel], None, [256], [0, 256])
            img_hist = cv2.calcHist([img], [channel], None, [256], [0, 256])
            img_hist = cv2.calcHist([img], [channel], None, [256], [0, 256])

            for name in ["ojama", "back"]:
                satu = np.mean(img[:, :, 1])
                if satu < 5 and name == "back":
                    diff = 1.00
                else:
                    template_img = self._field_template[name]
                    template_img_hist = cv2.calcHist(
                        [template_img], [channel], None, [256], [0, 256]
                    )
                    template_img_hist = cv2.calcHist(
                        [template_img], [channel], None, [256], [0, 256]
                    )
                    template_img_hist = cv2.calcHist(
                        [template_img], [channel], None, [256], [0, 256]
                    )
                    diff = cv2.compareHist(template_img_hist, img_hist, 0)
                    diff = cv2.compareHist(template_img_hist, img_hist, 0)
                    diff = cv2.compareHist(template_img_hist, img_hist, 0)

                differences.append(diff)
        puyo_type = differences.index(max(differences))
        return puyo_type


class FieldConstructor(object):
    def __init__(self, puyo_types):
        self._puyo_types = puyo_types
        self._field_template = {}
        for name in self._puyo_types:
            img = cv2.resize(cv2.imread(f"images/field/{name}.jpg"), (40, 40))
            self._field_template[name] = img

    def make_field_construct(self, field):
        init_img = np.zeros((480, 240, 3), dtype=np.uint8)
        for h in range(12):
            for w in range(6):
                puyo = field[h, w]
                puyo = self._puyo_types[int(puyo)]
                template_img = self._field_template[puyo]
                grid_h = h * 40
                grid_w = w * 40
                init_img[grid_h : grid_h + 40, grid_w : grid_w + 40] = template_img
        return init_img


def field_edit(field):
    if field[1, 2] == 6:
        field[0, 2] = 6
    return field


puyo_types = ["aka", "ao", "kiiro", "midori", "murasaki", "ojama", "back"]
classifier = puyo_classifier(puyo_types)
NEXT_LABELS = 5
FIELD_LABELS = 7
import time
def main():
    img = cv2.imread("banmen3.png")
    #img = cv2.convertScaleAbs(img, alpha=0.8, beta = -30)
    #th, img = cv2.threshold(img, 111, 255, cv2.THRESH_BINARY)
    fields = collections.deque([], 2) 
    start = time.time()
    field_puyos = get_field_info(img)
    print(time.time() - start)
    one_hot_field = np.array(np.eye(FIELD_LABELS)[field_puyos])
    fields.append(one_hot_field)
    fields.append(one_hot_field)
    #print(fields[0])
    #print(fields[1])
    #print(one_hot_field.ndim)
    print([len(v) for v in field_puyos])
    print(one_hot_field.shape)
    print(one_hot_field[0].shape)
    #next_puyos = get_next_puyo_info(img)
    #end = time.time()
    #print(next_puyos)
    #one_hot_next = np.array(np.eye(NEXT_LABELS)[next_puyos])
    #print(one_hot_next.shape)
    #print(one_hot_next[0].reshape(1,2,5))
    #print(one_hot_next[1].reshape(1,2,5))

    fc = FieldConstructor(puyo_types)
    player1_img = fc.make_field_construct(field_puyos[0])
    cv2.imwrite("player1_img.jpg", player1_img)
    player2_img = fc.make_field_construct(field_puyos[1])
    cv2.imwrite("player2_img.jpg", player2_img)

    '''
    for i in range(2):
        for j in range(12):
            for k in range(6):
                if field_puyos[i][j][k] == 6:
                    print("Yes")
    '''

if __name__ == "__main__":
    main()