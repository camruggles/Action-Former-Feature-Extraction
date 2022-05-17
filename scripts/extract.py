import os
import datetime
import sys
import time
import pdb
import torch
import torchvision
import numpy  as np
from my import I3D_BackBone

# the location of the files and features to be output
dirpath = './thumosextraction/validation/' # the location of the videos
nppath = './features/' # the location where the features are to be output

file1 = "./list.txt" # the list of files that have completed extraction
file2 = './todo.txt' # the list of all files to undergo feature extraction



# the list of files that have undergone extraction
f = open(file1, "r")
done = f.read().split("\n")

# the list of all files
f2 = open(file2, "r")
entries = f2.read().split("\n")

# prints how many files have been extracted
for z in range(len(entries)):
    # print remaining files and whether or not they have been extracted
    if (entries[z][:-4]+'.npy') in done:
      print(z, entries[z], 'extracted')
    else:
      print(z, entries[z])


def process(V):

    crop = torchvision.transforms.CenterCrop(224)
    res = torchvision.transforms.Resize((256,455))
    # change video to be (numclips, 3,16,224,224)
    a2 = crop(res(V/255.0))
    return a2


# this is a helper function to divide a video into increments of 100
def getBounds(n):
  arr = []
  i=0
  while i < n:
    arr.append(i)
    i += 100
  arr.append(n)
  return arr




# initialize extractor
model = I3D_BackBone(final_endpoint='Logits')
# load weights
model.load_pretrained_weight(model_path="./rgb_imagenet.pt")
# set to eval
model.eval()
# send model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)



'''
read in entire video, and count the number of clips
load 100 clips at a time,
extract clips,
save features in a vector
save vector to output file
'''

with torch.no_grad():
  for ent in entries:
    # don't extract a file that's already been done
    if (ent[:-4]+'.npy') in done: continue


    t1 = time.time()
    path = dirpath+ent # get a video path
    print(ent)

    # read the video time stamps to load the video into chunks, loading a whole video could be too costly
    stamps, _ = torchvision.io.read_video_timestamps(path, 'sec')
    print(len(stamps))

    # this line of code can be used to prioritize smaller videos first if desired
    #if len(stamps) > 10000: continue
    # 

    # read in video and reshape, compute length of video
    video = torchvision.io.read_video(path)[0]
    video = video.permute(3,0,1,2)
    length = video.shape[1]

    # separate ent to have the mp4 extension
    filename = ent[:-4]


    i = 0
    N = 0
    # count the number of clips that will be computed
    while i + 16 <= length:
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
        print(i2, '/', N) # print progress

        # prepare clips for extraction
        clips = []
        for i in range(i1,i2): # extract features for 100 clips at a time
            clip = process(video[:,(i)*4:(i)*4+16, :, :]) # extract a clip
            clips.append(clip)
        clipstack = torch.stack(clips)
        clipstack = clipstack.to(device)

        out = model(clipstack)
        output[i1:i2] = out.squeeze().cpu().numpy()

    # save to numpy file
    outfile = nppath+filename
    np.save(outfile, output)


    # print time spent extracting video
    # also prints wall clock if desired

    print(outfile, output.shape, time.time()-t1)
    print(datetime.datetime.now())
    print("="*10)
    # if you pipe output to file, i.e. python extract.py > output.txt
    # then this line will ensure you can see updates, and that the print function isn't buffering
    sys.stdout.flush()



