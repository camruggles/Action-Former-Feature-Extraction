import os
import datetime
import sys
import time
import pdb
import torch
import torchvision
import numpy  as np


from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale
from torch.nn import Identity, Module
from torchvision.transforms import Compose, Lambda, TenCrop
from torchvision.transforms._transforms_video import  CenterCropVideo, NormalizeVideo

import random


device=torch.device("cuda:0")

stackincrement = 80
# the list of files that have undergone extraction




transforms = [
    # Lambda(lambda x: x[:, ::config.dilation]),
    Lambda(lambda x: x / 255.0),
    NormalizeVideo([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ShortSideScale(256),
    CenterCropVideo(224),
]
transform = Compose(transforms)

def process(V):
    # V = V.permute(3,0,1,2)
    V = V.float()
    # print(V.shape)
    V = transform(V)
    return V

# this is a helper function to divide a video into increments of 100
def getBounds(n):
  arr = []
  i=0
  while i < n:
    arr.append(i)
    i += stackincrement
  arr.append(n)
  return arr





'''
read in entire video, and count the number of clips
load 100 clips at a time,
extract clips,
save features in a vector
save vector to output file
'''
def extract_feats(model, dirpath, ent, tempinfo=None):
    model.eval()
    with torch.no_grad():

        t1 = time.time()
        path = os.path.join(dirpath,ent) # get a video path
        # print(ent)

        # read the video time stamps to load the video into chunks, loading a whole video could be too costly
        large=False
        if large or tempinfo is not None:
          # stamppath = os.path.join(dirpath, ent[:-4] + "_stamps.npy")
          stamps, _ = torchvision.io.read_video_timestamps(path, 'pts')
          # stamps = np.load(stamppath)
        #   print(len(stamps))

        # this line of code can be used to prioritize smaller videos first if desired
        #if len(stamps) > 10000: continue
        # 

        # read in video and reshape, compute length of video
        # print(path)
        if tempinfo is not None:
          st,ed = tempinfo
          st = st*4
          ed = ed*4 + 15
          # print(path, st, ed, len(stamps), tempinfo)
          temp_diff = ed-(len(stamps)-1)
          ed = min(ed,len(stamps)-1)
          # print(path, st, ed, len(stamps), tempinfo)
          # print(stamps[st], stamps[ed])
          video = torchvision.io.read_video(path, stamps[st], stamps[ed], 'pts')[0]
          video = video.permute(3,0,1,2)
          if temp_diff > 0:
            # print("temp_diff > 0")
            last_frame = video[:,-1,:,:].unsqueeze(1)
            for i in range(temp_diff):
              # print(video.shape, last_frame.shape)
              video = torch.cat([video, last_frame],dim=1)
              # print(video.shape, last_frame.shape)

          length = video.shape[1]

        elif not large:
          video = torchvision.io.read_video(path)[0]
          video = video.permute(3,0,1,2)
          length = video.shape[1]
        else:
          length = len(stamps)

        # separate ent to have the mp4 extension
        filename = ent[:-4]


        i = 0
        N = 0
        # count the number of clips that will be computed
        while i + 16 < length:
            N += 1
            i += 4
        '''
        while i+16 <= length:
            clip = video[:, i:i+16, :, :]
            clips.append(clip)
            i = i+4
        '''


        bounds = getBounds(N) # get 100 count increments of the number of clips
        output=np.empty((N, 1024)) # feature vector to be saved

        # extract features in 100 clip increments
        for j in range(len(bounds)-1):
            i1 = bounds[j]
            i2 = bounds[j+1]
            # print(i2, '/', N, end=",") # print progress

            # prepare clips for extraction
            clips = []
            if large:
              video = torchvision.io.read_video(path, stamps[i1], stamps[(i2-1)*4+16], 'pts')[0]
              video = video.permute(3,0,1,2)
              # print(video.shape)
            for i in range(i1,i2): # extract features for 100 clips at a time
                if not large:
                  clip = process(video[:,(i)*4:(i)*4+16, :, :]) # extract a clip
                  clips.append(clip)
                else:
                  clip = video[:,(i-i1)*4:(i-i1)*4+16, :, :]
                  # print(clip.shape)
                  # print((i-i1)*4)
                  # print((i-i1)*4+16)
                  clip = process(clip)
                  clips.append(clip)
            clipstack = torch.stack(clips)
            clipstack = clipstack.to(device)

            out = model(clipstack)
            output[i1:i2, :] = out.squeeze().cpu().numpy()
    model.train()
    # print()
    return output


def extract_video(model, vid_folder, feat_folder):
    model.eval()
    files = os.listdir(feat_folder)
    for f in files:
        r = random.uniform(0,1)
        # print(r)
        if r < -1.0:
            print("updating", f)
            feat_file = f
            vid_file = f[:-4] + ".mp4"
            output = extract_feats(model, vid_folder, vid_file)

            feats = np.load(os.path.join(feat_folder, feat_file))

            m1 = output.shape[0]
            m2 = feats.shape[0]

            assert abs(m1-m2) <= 1
            m = min(m1,m2)
            # print(m1,m2,m)

            feats[:, :1024] = output
            np.save(os.path.join(feat_folder, feat_file), feats)


    model.train()

def extract_raw(model, dirpath, ent, tempinfo):
    model.eval()
    with torch.no_grad():

        t1 = time.time()
        path = os.path.join(dirpath,ent) # get a video path
        # print(ent)

        st,ed = tempinfo
        st = st*4
        ed = ed*4 + 15
        
        stamps, _ = torchvision.io.read_video_timestamps(path, 'pts')


        temp_diff = ed-(len(stamps)-1)
        ed = min(ed,len(stamps)-1)

        video = torchvision.io.read_video(path, stamps[st], stamps[ed], 'pts')[0]
        video = video.permute(3,0,1,2)


        if temp_diff > 0:
          # print("temp_diff > 0")
          last_frame = video[:,-1,:,:].unsqueeze(1)
          for i in range(temp_diff):
            # print(video.shape, last_frame.shape)
            video = torch.cat([video, last_frame], dim=1)
            # print(video.shape, last_frame.shape)

        '''

          st,ed = tempinfo
          st = st*4
          ed = ed*4 + 15
          print(path, st, ed, len(stamps), tempinfo)
          print(stamps[st], stamps[ed])
          video = torchvision.io.read_video(path, stamps[st], stamps[ed], 'pts')[0]
          video = video.permute(3,0,1,2)
          if temp_diff > 0:
            last_frame = video[-1,:,:,:]
            for i in range(temp_diff):
              video = torch.cat([video, last_frame])

        '''
        length = video.shape[1]


        # separate ent to have the mp4 extension
        filename = ent[:-4]


        i = 0
        N = 0
        # count the number of clips that will be computed
        clips = []
        while i*4 + 16 < length:
          v = video[:,(i)*4:(i)*4+16, :, :]
          # print(i,v.shape,st,ed, i*4, i*4+16, length, v.device)
          clip = process(v) # extract a clip
          clips.append(clip)
          i += 1
        clipstack = torch.stack(clips)
    # pdb.set_trace()
    model.train()
    return clipstack

