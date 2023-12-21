import numpy as np
import random

FIELD_LABELS = 7
field = np.zeros((2,12,6), dtype=np.uint8)
for i in range(2):
    for j in range(12):
        for k in range(6):
            field[i][j][k] = random.randint(0,6)

fieldlist = []
fieldlist.append(field)
one_hot_field = np.array(np.eye(FIELD_LABELS)[fieldlist[0]])
print(one_hot_field.shape)
one_hot_field[0].reshape(1,12,6,7)
print(one_hot_field[0].reshape(1,12,6,7).shape)
