import csv
import matplotlib.pyplot as plt

#csvファイルを指定
MyPath = './reward_500.csv'

#csvファイルを読み込み
rows = []
with open(MyPath) as f:
    reader = csv.reader(f)
    for row in reader:
        rows.append(row)

float_list = []

for r in rows:
    float_list.append(float(r[0]))

#print(float_list)
plt.plot(float_list)
plt.xlabel("エピソード数",fontname="MS Gothic")
plt.ylabel("報酬",fontname="MS Gothic")
plt.show()