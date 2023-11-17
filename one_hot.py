import numpy as np

target_vector = [[1,2,3],[3,2,1]]
target_one_hot =[]

n_labels = 4
#for i in range(2):
target_one_hot.append(np.eye(n_labels)[target_vector])
    

print(target_one_hot)
print(type(target_one_hot))
