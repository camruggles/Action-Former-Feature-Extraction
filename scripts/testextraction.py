import os
import datetime
import sys
import time
import pdb
import torch
import torchvision
import torch.nn as nn
import numpy  as np
from i3d import I3D_BackBone

# set paths
dirpath = './thumosextraction/test/' # location of test videos
nppath = './testfeatures/' # output directory
featurepath = '/home/cruggles/temporary/thumos/i3d_features/' # location of test features (need to merge with optical flow)
file1 = "list.txt" # list of extracted features
file2 = "todo.txt" # list of filenames of test feature filenames
model_path="/home/cruggles/rgb_imagenet.pt"

# get all the completed files
f = open(file1, "r")
done = f.read().split("\n")

# get all the remaining files
f2 = open(file2, "r")
entries = f2.read().split("\n")


# prints how many files have been extracted
for z in range(len(entries)):
    if (entries[z][:-4]+'.npy') in done:
      print(z, entries[z], 'extracted')
    else:
      print(z, entries[z])

# process videos
def process(V):

    crop = torchvision.transforms.CenterCrop(224)
    res = torchvision.transforms.Resize((256,455))
    # change video to be (numclips, 3,16,224,224)
    a2 = crop(res(V/255.0))
    return a2

# get 100 spaced markings
def getBounds(n):
  arr = []
  i=0
  while i < n:
    arr.append(i)
    i += 250
  arr.append(n)
  return arr




# initialize extractor
model = I3D_BackBone(final_endpoint='Logits')
# load weights
model.load_pretrained_weight(model_path=model_path)
model = nn.DataParallel(model, device_ids=[0,1])
# set to eval
model.eval()
# send model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)





with torch.no_grad():
  # iterate over entries and read in the video
  for ent in entries:
    if (ent[:-4]+'.npy') in done: continue
    t1 = time.time()

    # get the filename
    path = dirpath+ent
    print(ent)
    filename = ent[:-4]


    # read the video from the file path

    # get the timestamps

    # there was a glitch with some files where timestamps couldn't be loaded, so they were handledd differently
    stampsflag = True
    stamps = np.load('./teststamps/' + filename + '_stamps.npy')
    print(len(stamps))


    if len(stamps) != 0:
      # save the time stamps to a file, in case they're needed at some point
      #np.save("/home/cruggles/stamps/test/" + ent[:-4], np.array(stamps, dtype = np.int64))
      #if len(stamps) > 10000: continue
      length = len(stamps)#video.shape[1]
    else:
      stampsflag=False
      video = torchvision.io.read_video(path)[0]
      video = video.permute(3,0,1,2)
      print(video.shape)
      length = video.shape[1]
    print(stampsflag)
    print(length)


    # load up features
    feats = np.load(featurepath + ent[:-4]  + '.npy')

    # separate ent to have the mp4 extension
    filename = ent[:-4]

    # find the number of clips that will be present

    i = 0
    N = 0
    # the presence of timestamps effects how the clips are counted
    while i + 16 <= length and not stampsflag:
        N += 1
        i += 4

    while i + 15 < length and stampsflag:
        N += 1
        i += 4
    '''
    while i+16 <= length:
        clip = video[:, i:i+16, :, :]
        clips.append(clip)
        i = i+4
    '''

    # get 100 incremental bounds
    N = min(N, feats.shape[0])
    bounds = getBounds(N)
    output=np.empty((N, 1024))

    # extract features in increments of 100
    for j in range(len(bounds)-1):
        i1 = bounds[j]
        i2 = bounds[j+1]
        print(i2, '/', N)
        clips = []

        # extract clips
        if stampsflag:
           # extract a portion of the video, some videos are too large, so this relieves memory
           video = torchvision.io.read_video(path, stamps[i1*4], stamps[(i2-1)*4+15], 'pts')[0].permute(3,0,1,2)
           video = process(video)
        for i in range(i1,i2):
            # reload video
            if stampsflag: # indicates that stamps are being used
                i = i-i1 # indices are off if you use timestamps to load segments of video
                v2 = video[:, i*4:i*4+16, :, :]
            if not stampsflag:
                v2 = video[:, i*4:i*4+16, :, :]
            clip = process(v2)
            #print(clip.shape)
            clips.append(clip)
        clipstack = torch.stack(clips)
        clipstack = clipstack.to(device)
        # extract features
        out = model(clipstack)
        output[i1:i2] = out.squeeze().cpu().numpy()

    '''
    # extract features
    clipstack = torch.stack(clips)
    # send input to device
    print(clipstack.shape)
    #output = output.to(device)
    for i in range(len(bounds)-1):
        ins = clipstack[i1:i2, :]
        ins= ins.to(device)
        out = model(ins)
        output[i1:i2, :] = out.squeeze().cpu().numpy()
    '''


    # save to numpy file
    # save to features file
    # get feature file
    print(feats.shape)
    print(output.shape)
    feats[:, :1024] = output

    outfile = nppath+filename

    print(outfile, output.shape, time.time()-t1)
    np.save(outfile, feats)

    print(datetime.datetime.now())
    print("="*10)
    #del video, clips, clipstack, out, output, bounds
    sys.stdout.flush()



