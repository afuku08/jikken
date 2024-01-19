import csv

list = []

for i in range(10):
    list.append(i)

csv_path = r"./test.csv"

print(list)

with open(csv_path, 'w', newline='') as file:
    writer = csv.writer(file)
    for i in list:
        writer.writerow([i])