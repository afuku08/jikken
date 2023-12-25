import numpy as np
dodai = ["gtr", "ngtr", "yayoi", "da"]

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
    ruiji = 0
    for dodai in dodailist:
        tmp = np.count_nonzero(banmen == dodai) / dodai.size
        print(str(tmp))
        ruiji = max(ruiji, tmp)
    
    return ruiji


dodailist = []

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

import time
start = time.time()
get_dodai_reward(np.zeros((4,6)))
print(time.time() - start)

