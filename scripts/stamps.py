
import torch
import torchvision
import time
import numpy as np
import torchvision
import torch
from multiprocessing.pool  import ThreadPool
import datetime
from multiprocessing import Pool
import time


from multiprocessing import Process, Queue

savepath="./teststamps/"
videos_path = "./thumosextraction/test/"
videos_list_file = "testvids.txt"
print(datetime.datetime.now())

with open(videos_list_file) as f:
    lines = [line.rstrip() for line in f]
print(lines)

def loadStamps(filename):
    #filename = 'video_validation_0000851.mp4'
    fname = videos_path + filename
    print(fname)
    t1 = time.time()
    stamps,_=torchvision.io.read_video_timestamps(fname, 'pts')
    f2 = savepath + filename[:-4] + "_stamps.npy"
    a=np.array(stamps, dtype=np.int64)
    np.save(f2, a)
    print(f2,a.shape, time.time()-t1, datetime.datetime.now())

p = Pool(8)
handler = p.map_async(loadStamps, lines)
results=handler.get()
p.close()
p.join()
print(datetime.datetime.now())
print('all done')

