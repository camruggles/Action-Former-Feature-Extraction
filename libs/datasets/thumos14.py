import os
import time
import copy
import json
import numpy as np
import random
import torchvision
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats

import torchvision
import torch

from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale
from torch.nn import Identity, Module
from torchvision.transforms import Compose, Lambda, TenCrop
from torchvision.transforms._transforms_video import  CenterCropVideo, NormalizeVideo

# from i3d import I3D
# from i3d2 import I3D_BackBone


transforms = [
    # Lambda(lambda x: x[:, ::config.dilation]),
    Lambda(lambda x: x / 255.0),
    NormalizeVideo([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ShortSideScale(256),
    CenterCropVideo(224),
]
transform = Compose(transforms)

def process(V):
    V = V.permute(3,0,1,2)
    V = V.float()
    V = transform(V)
    return V
    # crop = torchvision.transforms.CenterCrop(224)
    # res = torchvision.transforms.Resize((256,455))
    # if V.shape[0] > 500:
    #   # permute video to be of shape (3,16,224,224) after operations
    #   V = V.permute(3,0,1,2)
    #   V2 = torch.empty(V.shape[0], V.shape[1], 224, 224)
    #   for i in range(0, V.shape[1], 500): # process only 500 at a time for memory purposes
    #       tmp = V[:,i:min(i+500, V.shape[1]), :, :]
    #       # perform center crop and resizing
    #       #   tmp = crop(res(tmp/255.0)) # 
    #       tmp = res(tmp)
    #       tmp = tmp.double()
    #       tmp = tmp * 2 / 255 - 1
    #       tmp = crop(tmp)
    #       tmp = tmp.float()
    #       V2[:, i:min(i+500, V.shape[1]), :, :] = tmp
    #   a2 = V2

    # else: # for videos of less than 500 frames, just do this instead
    #   V = V.permute(3,0,1,2)
    #   #a2 = crop(res(V/255.0))
    #   V = res(V)
    #   V = V.double()
    #   V = V * 2 / 255 - 1
    #   V = crop(V)
    #   V = V.float()
    #   a2=V
    # return a2

@register_dataset("thumos")
class THUMOS14Dataset(Dataset):
    def __init__(
        self,
        is_training,     # if in training mode
        split,           # split, a tuple/list allowing concat of subsets
        feat_folder,     # folder for features
        json_file,       # json file for annotations
        feat_stride,     # temporal stride of the feats
        num_frames,      # number of frames for each feat
        default_fps,     # default fps
        downsample_rate, # downsample rate for feats
        max_seq_len,     # maximum sequence length during training
        trunc_thresh,    # threshold for truncate an action segment
        crop_ratio,      # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,       # input feat dim
        num_classes,     # number of action categories
        file_prefix,     # feature file prefix if any
        file_ext,        # feature file extension if any
        force_upsampling, # force to upsample to max_seq_len
        val_path,
        stampsfolder,
        num_clips_updated,
        is_resume=False
    ):
        # file path
        assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.json_file = json_file

        # split / training mode
        self.split = split
        self.is_training = is_training
        self.is_resume = is_resume

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio

        # load database and select the subset
        dict_db, label_dict = self._load_json_db(self.json_file)
        assert len(label_dict) == num_classes
        self.data_list = dict_db
        self.label_dict = label_dict

        # feature extractor folders
        self.val_path = val_path
        self.stampsfolder = stampsfolder
        self.num_clips_updated = num_clips_updated
        
        with open(self.json_file) as f: # annotations sampling
            data = json.load(f) # annotations sampling
        self.cameron_database = data['database'] # annotations sampling

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'thumos-14',
            'tiou_thresholds': np.linspace(0.3, 0.7, 5),
            # we will mask out cliff diving
            'empty_label_ids': [4],
        }

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']

        # if label_dict is not available
        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                for act in value['annotations']:
                    label_dict[act['label']] = act['label_id']

        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_db.items():
            # skip the video if not in the split
            if value['subset'].lower() not in self.split:
                continue
            # or does not have the feature file
            feat_file = os.path.join(self.feat_folder,
                                     self.file_prefix + key + self.file_ext)
            if not os.path.exists(feat_file):
                continue

            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."

            # get video duration if available
            if 'duration' in value:
                duration = value['duration']
            else:
                duration = 1e8

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                # a fun fact of THUMOS: cliffdiving (4) is a subset of diving (7)
                # we remove all cliffdiving from training and output 0 at inferenece
                # as our model can't assign two labels to the same segment
                segments, labels = [], []
                for act in value['annotations']:
                    if act['label_id'] != 4:
                        segments.append(act['segment'])
                        labels.append([label_dict[act['label']]])

                segments = np.asarray(segments, dtype=np.float32)
                labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
            else:
                segments = None
                labels = None
            dict_db += ({'id': key,
                         'fps' : fps,
                         'duration' : duration,
                         'segments' : segments,
                         'labels' : labels
            }, )

        return dict_db, label_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]
        #print(video_item)

        # load features
        filename = os.path.join(self.feat_folder,
                                self.file_prefix + video_item['id'] + self.file_ext)
        feats = np.load(filename).astype(np.float32)

        # deal with downsampling (= increased feat stride)
        feats = feats[::self.downsample_rate, :]
        feat_stride = self.feat_stride * self.downsample_rate
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))
        # if ("158" in video_item['id']):
        #     print("feats1")
        #     print(feats.shape)
        #     print(feats[:10, 0].T)
        #     print(feats[:10, 205:209].T)
        #     print(feats[:10, 205:209].T.shape)

        C,T = feats.shape
        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                (video_item['segments'] * video_item['fps'] - 0.5 * self.num_frames) / feat_stride
            )
            labels = torch.from_numpy(video_item['labels'])
        else:
            segments, labels = None, None

        # return a data dict
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : video_item['fps'],
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'full_feats'      : copy.deepcopy(feats), # TODO check the copy operation
                     'tempinfo'        : (0,T), # used to mark which frames are having a feature update
                     'feat_num_frames' : self.num_frames}
        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, self.crop_ratio
            )

        # TODO load a chunk of the raw video, store the frames and the start/end points, incorporate downsamnpling
        val_path = self.val_path
        stampsfolder = self.stampsfolder
        stampspath = os.path.join(stampsfolder, video_item['id'] + '_stamps.npy')  # import video metadata to speed up video loading

        if self.is_training:
          filename = os.path.join(val_path, self.file_prefix + video_item['id'] + ".mp4") 

        contiguous_size=4
        def get_feats(frames, start,end):

            def has_overlap(t1, t2):
                f1,f2 = t1
                i1,i2 = t2

                if i1 <= f2 and i1 >= f1:
                    return True
                if i2 <= f2 and i2 >= f1:
                    return True
                if f1 <= i2 and f1 >= i1:
                    return True
                if f2 <= i2 and f2 >= i1:
                    return True
                return False

            final_list = []
            f1 = frames[0]
            f2 = frames[1]
            f1 = int(f1)
            f2 = int(f2)

            for i in range(start,end-contiguous_size-1, contiguous_size):
                i1 = i*4
                i2 = i1+16

                if has_overlap((f1,f2), (i1,i2)):
                    final_list.append(i)
            return final_list

        # get metainfo
        C,T = feats.shape
        (start_video,end_video) = data_dict['tempinfo'] # start and endpoints of trunc_feats subset from feature file
        num_clips_updated = self.num_clips_updated # the number M out of T clip features to update
        self.action_sampling=False
        if self.action_sampling:
            # print(data_dict['video_id'])
            annos = self.cameron_database[data_dict['video_id']]['annotations']
            feats_idx = []
            for item in annos:
                frames = item['segment(frames)']
                sub_feats_list = get_feats(frames,start_video, end_video)
                feats_idx.extend(sub_feats_list)
            feats_idx = list(set(feats_idx))
            feats_idx.sort()
            sampling_options=feats_idx
            # print(len(sampling_options))
            if len(sampling_options) < 40:
                sampling_options = [x for x in range(start_video, end_video-contiguous_size, contiguous_size)]
        else: 
            sampling_options = [x for x in range(start_video, end_video-contiguous_size, contiguous_size)]

        stratified_random=False
        if self.is_training:
            if stratified_random:
                j = 0
                feature_index = []
                for _ in range(num_clips_updated // contiguous_size):

                    # feature_sub_index = random.randint(start_video, end_video-1-contiguous_size)
                    feature_sub_index = random.choice(sampling_options)
                    
                    feature_sub_list = [x for x in range(feature_sub_index, feature_sub_index + contiguous_size)]

                    # flat_feature_list=[item for sublist in feature_index for item in sublist]
                    # while len(list(set(feature_sub_index) & set(flat_feature_list))) != 0:
                    #     feature_sub_index = random.randint(start_video, end_video-1-contiguous_size)
                    #     feature_sub_index = [x for x in range(feature_sub_index, feature_sub_index + contiguous_size)]
                    #     j += 1
                    #     if j >= 100:
                    #         stratified_random=False
                    #         print("set stratified random to false")
                    #         break
                    
                    feature_index.append(feature_sub_list)
                    sampling_options.remove(feature_sub_index)
                
                
                frame_indices = []
                for index_tmp in feature_index:
                    index = index_tmp[0]
                    point1 = index * feat_stride
                    point2 = (index + contiguous_size-1) * feat_stride + 16
                    frame_indices.append((point1,point2))
                # print(feature_index, frame_indices)
                assert len(feature_index) == len(frame_indices)
                
            if not stratified_random:
                # print(start_video, end_video, end_video-2-num_clips_updated, num_clips_updated)
                if end_video < num_clips_updated:
                    feature_index = start_video
                    num_clips_updated = end_video - start_video - 1
                else:
                    feature_index = random.randint(start_video, end_video-2-num_clips_updated) # location of updated features in feature vector
                frame_index=feature_index*feat_stride # location of updated features in video frames
                end_frame = frame_index+(num_clips_updated-1)*feat_stride+16
                # print(frame_index, feature_index, end_frame, num_clips_updated)
        
        

        # sample between st and ed since a subset of the full features are selected in truncate feats
        frames = []
        if self.is_training:
          stamps = np.load(stampspath)


        raw = [] # torch tensor of video frames
        fileframes = [] # torch tensor of indices where features in the .npy file will be updated
        
        get_raw_frames = False
        if self.is_training and not self.is_resume and get_raw_frames:
            assert not self.is_resume
            if not stratified_random:
                if self.is_training:
                    # extract frame locations using stamps vector
                    end_frame = frame_index+(num_clips_updated-1)*feat_stride+16
                    video = torchvision.io.read_video(filename, stamps[frame_index], stamps[end_frame], 'pts')[0] # note that read_video is inclusive of the second endpoint, so (M-1) is necessary
                    video = process(video) # resize and perform center crop divide by 255

                    for i in range(num_clips_updated): # check that all 16 frames are being loaded
                        v = video[:,(i*feat_stride):(i*feat_stride+16), :,:] # get the video frames
                        raw.append(v) # place them in update vector
                        frames.append(feature_index + i - start_video) # add index location where data dict is updated
                        fileframes.append(feature_index + i) # add index location where the FILE is updated
                        # note that the feature vector used by transformer is a subset of the features in the saved file
                        # so it's necessary to update both the file and the data dict
            if stratified_random:
                # print(filename)
                for i in range(len(feature_index)):
                    (st_idx, ed_idx) = frame_indices[i]
                    ft_idx = feature_index[i]
                    video = torchvision.io.read_video(filename, stamps[st_idx], stamps[ed_idx-1], 'pts')[0] # careful of -1 here
                    n_update_frames = (contiguous_size-1)*feat_stride + 16
                    video = video[0:n_update_frames, :, :,:]
                    # if "158" in video_item['id']:
                    #     print("vid1")
                    #     print(video.permute(3,0,1,2)[:2,:2,:2,:2])
                    # print(video.shape, stamps[st_idx], stamps[ed_idx-1], st_idx, ed_idx, ed_idx-1)
                    video = process(video)
                    # if "158" in video_item['id']:
                    #     print("vid1")
                    #     print(video[:2,:2,:2,:2])
                    # v = video[:,0:]
                    for j in range(contiguous_size):
                        v = video[:, (j*feat_stride):(j*feat_stride+16), :, :]
                        raw.append(v)
                        frames.append(ft_idx[0] - start_video + j)
                        fileframes.append(ft_idx[0] + j)
        
        data_dict['frames'] = frames
        data_dict['fileframes'] = fileframes
        if self.is_training and not self.is_resume and get_raw_frames:
          raw = torch.stack(raw)
          data_dict['raw'] = raw # add data dict item for use later
        # (frames)
        # print(filefprintrames)
        # print(frames,fileframes, data_dict['video_id'])

        # if ("158" in video_item['id']):
        #     print('feats2')
        #     print(feats.shape)
            # print(feats[0, 0:10].T)
            # print(feats[200:215, 0:10].T)

        return data_dict



