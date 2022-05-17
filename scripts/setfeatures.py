# setfeatures.py


import numpy as np
import os


neopath = "/home/cruggles/temporary/neothumos/i3d_features/" # location of where new features are to be aved
oldpath = "/home/cruggles/temporary/thumos/i3d_features/" # location of features downloaded from actionformer
path = "./features/" # location of fresh features
file1 = 'transfer.txt' # list of the filenames of features

f = open(file1, "r")
files = f.read().split("\n")

for filename in files:
    oldfile = os.path.join(oldpath, filename)
    halffile = os.path.join(path, filename)
    savefile = os.path.join(neopath, filename)


    oldfeats = np.load(oldfile)
    feats= np.load(halffile)
    m = feats.shape[0]
    n = oldfeats.shape[0]
    m = min(m,n)
    print(oldfeats.shape, feats.shape)
    oldfeats[:m, :1024] = feats[:m, :]
    np.save(savefile,oldfeats)




