# from random import randint, expovariate

from environments.vne_env import VNR

# TIME_STEP_SCALE = 1 / 10
# for _ in range(100):
#     print(int(expovariate(0.002 * (1.0 / TIME_STEP_SCALE))))


import heapq

from common import config

heap = []

for vnr_id in range(10):
    vnr = VNR(
        id=vnr_id,
        vnr_duration_mean_rate=config.VNR_DURATION_MEAN_RATE,
        delay=config.VNR_DELAY,
        time_step_arrival=0
    )

    heapq.heappush(heap, vnr)

for _ in range(len(heap)):
    print(heapq.heappop(heap))