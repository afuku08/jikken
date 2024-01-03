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
                puyo = classifier.predict(grid, template_type="field")
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
        for i in range(11):
            for j in range(6):
                if init_field[i+1][j] == 0:
                    init_field[i][j] = 0

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
    for i in range(2):
        result = player1_next[i] not in puyo_cont
        if len(puyo_cont) == 4:
            if result:
                player1_next[i] = 0
            else:
                if result:
                    puyo_cont.append(player1_next[i])
                        
        player1_next[i] = puyo_cont.index(player1_next[i])

    player1_next_next = img[132 : 172 , 259 : 274]
    h, w, c = player1_next_next.shape
    player1_next_next = player1_next_next[: h // 2], player1_next_next[h // 2 :]
    player1_next_next = [
        classifier.predict(i, template_type="p1") for i in player1_next_next
    ]
    for i in range(2):
        result = player1_next_next[i] not in puyo_cont
        if len(puyo_cont) == 4:
            if result:
                player1_next_next[i] = 0
            else:
                if result:
                    puyo_cont.append(player1_next_next[i])
                        
        player1_next_next[i] = puyo_cont.index(player1_next_next[i])

    player1_nexts = [player1_next, player1_next_next]
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

        channel_b = 0
        channel_g = 1

        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hist_b = cv2.calcHist([img], [channel_b], None, [256], [0, 256])
        img_hist_g = cv2.calcHist([img], [channel_g], None, [256], [0, 256])

        for name in self._puyo_types[:-2]:
            if template_type == "field":
                template_img = self._field_template[name]
            elif template_type == "p1":
                template_img = self._p1_template[name]
            elif template_type == "p2":
                template_img = self._p2_template[name]
            template_img_hist_b = cv2.calcHist(
                [template_img], [channel_b], None, [256], [0, 256]
            )
            template_img_hist_g = cv2.calcHist(
                [template_img], [channel_g], None, [256], [0, 256]
            )

            diff = cv2.compareHist(template_img_hist_b, img_hist_b, 0)
            diff += cv2.compareHist(template_img_hist_g, img_hist_g, 0)
            diff /= 2
            differences.append(diff)

        if template_type == "field":
            channel = 0
            img_hist_b = cv2.calcHist([img], [channel_b], None, [256], [0, 256])
            img_hist_g = cv2.calcHist([img], [channel_g], None, [256], [0, 256])

            for name in ["ojama", "back"]:
                satu = np.mean(img[:, :, 1])
                if satu < 5 and name == "back":
                    diff = 1.00
                else:
                    template_img = self._field_template[name]
                    template_img_hist_b = cv2.calcHist(
                        [template_img], [channel_b], None, [256], [0, 256]
                    )
                    template_img_hist_g = cv2.calcHist(
                        [template_img], [channel_g], None, [256], [0, 256]
                    )
                    diff = cv2.compareHist(template_img_hist_b, img_hist_b, 0)
                    diff += cv2.compareHist(template_img_hist_g, img_hist_g, 0)
                    diff /= 2

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


dodailist = []
def read_dodai():
    dodai = ["gtr", "ngtr", "yayoi", "da"]

    for name in dodai:
        f = open('./dodai/%s.csv' % name, 'r')
        data = f.read()
        data = data.replace("\n", "")
        test_str = list(data)
        test_str = np.array(test_str)
        test_str = test_str.reshape(4,6)
        test_str = change_int(test_str)
        dodailist.append(test_str)
        for i in range(3):
            test_str = change_color(test_str)
            dodailist.append(test_str)
        f.close()

def change_color(banmen):
    for i in range(4):
        for j in range(6):
            now = banmen[i][j]
            if now == 0:
                continue
            now += 1
            if now == 5:
                now = 1
            banmen[i][j] = now
    return banmen

def change_int(banmen):
    ban = np.zeros((4,6))
    for i in range(4):
        for j in range(6):
            ban[i][j] = int(banmen[i][j])

    return ban

def get_dodai_reward(banmen):
    banmen = np.array(banmen)
    print(banmen)
    ruiji = 0
    for dodai in dodailist:
        tmp = np.count_nonzero(banmen == dodai) / dodai.size
        #print(str(tmp))
        ruiji = max(ruiji, tmp)
    
    return ruiji


puyo_types = ["aka", "ao", "kiiro", "midori", "murasaki", "ojama", "back"]
classifier = puyo_classifier(puyo_types)
NEXT_LABELS = 4
FIELD_LABELS = 6
import time
def main():
    img = cv2.imread("banmen3.png")
    fields = collections.deque([], 2) 
    start = time.time()
    field_puyos = get_field_info(img)
    #print(time.time() - start)
    #print(field_puyos)
    #print(np.unique(field_puyos))
    #print(field_puyos[0][8:])
    read_dodai()
    reward = get_dodai_reward(field_puyos[0][8:])
    #print(reward)
    one_hot_field = np.array(np.eye(FIELD_LABELS)[field_puyos])
    fields.append(one_hot_field)
    fields.append(one_hot_field)
    next_puyos = get_next_puyo_info(img)
    one_hot_next = np.array(np.eye(NEXT_LABELS)[next_puyos])
    nexts = collections.deque([], 2)
    nexts.append(one_hot_next)
    nexts.append(one_hot_next)
    print(type(nexts[0][0]))
    print(nexts[0][0].reshape(1,2,4))
    print(nexts[0][1].reshape(1,2,4))

    fc = FieldConstructor(puyo_types)
    player1_img = fc.make_field_construct(field_puyos[0])
    cv2.imwrite("player1_img.jpg", player1_img)
    player2_img = fc.make_field_construct(field_puyos[1])
    cv2.imwrite("player2_img.jpg", player2_img)

if __name__ == "__main__":
    main()