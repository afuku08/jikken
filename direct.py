import pydirectinput as direct

direct.PAUSE = 0.01
import time
start = time.time()
direct.press("a")
print(time.time() - start)