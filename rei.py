H = 324
W = 132

h_unit = H // 12
w_unit = W // 6    
        
count = 0
for h in range(0, H, h_unit):
    for w in range(0, W, w_unit):
        print(str(h) + ":" + str(h+h_unit) + " " + str(w) + ":" + str(w+w_unit))
        count += 1

print(count)