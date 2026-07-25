import torch
import time
timings = {}
enable_timing = True

def start(name):
    if enable_timing:
        if not name in timings:
            timings[name] = 0
        torch.cuda.synchronize()
        start_time = time.time()
        timings[name] -= start_time

def stop(name):
    if enable_timing:
        torch.cuda.synchronize()
        end_time = time.time()
        timings[name] += end_time
