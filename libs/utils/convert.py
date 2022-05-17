#run.py
import torch
import pdb 
import numpy as np
import torchvision
import cv2 
from my import I3D_BackBone
import time
import numpy as np
import tensorflow as tf




def convert_model(model):


    CHECKPOINT_PATHS = {
        'rgb': '/home/cruggles/actionformer_release-main/libs/modeling/checkpoints/rgb_scratch/model.ckpt',
        'rgb600': '/home/cruggles/actionformer_release-main/libs/modeling/checkpoints/rgb_scratch_kin600/model.ckpt',
        'flow': '/home/cruggles/actionformer_release-main/libs/modeling/checkpoints/flow_scratch/model.ckpt',
        'rgb_imagenet': '/home/cruggles/actionformer_release-main/libs/modeling/checkpoints/rgb_imagenet/model.ckpt',
        'flow_imagenet': '/home/cruggles/actionformer_release-main/libs/modeling/checkpoints/flow_imagenet/model.ckpt'
   }



    file_name = CHECKPOINT_PATHS['rgb600']

    from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
    from tensorflow.python.training import py_checkpoint_reader
    #print_tensors_in_checkpoint_file(file_name=file_name, tensor_name='', all_tensors=False, all_tensor_names=True)
    # conversion to torch
    reader = py_checkpoint_reader.NewCheckpointReader(file_name)
    state_dict = {
        v: reader.get_tensor(v) for v in reader.get_variable_to_shape_map()
    }




    def convert(state_dict, st):
        T = state_dict[st]
        T = torch.from_numpy(T)
        T = T.permute(4,3,2,1,0)
        if "beta" in st:
            N = T.shape[0]
            T = T.reshape(N)
        return T





    # model = I3D_BackBone(final_endpoint='Logits')



    # start here?
    st = 'Conv3d_1a_7x7/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Conv3d_1a_7x7.conv3d.weight.data.shape == T.shape:
        model._model.Conv3d_1a_7x7.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Conv3d_1a_7x7/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Conv3d_1a_7x7.bn.bias.data.shape == T.shape:
        model._model.Conv3d_1a_7x7.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Conv3d_2b_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Conv3d_2b_1x1.conv3d.weight.data.shape == T.shape:
        model._model.Conv3d_2b_1x1.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Conv3d_2b_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Conv3d_2b_1x1.bn.bias.data.shape == T.shape:
        model._model.Conv3d_2b_1x1.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Conv3d_2c_3x3/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Conv3d_2c_3x3.conv3d.weight.data.shape == T.shape:
        model._model.Conv3d_2c_3x3.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Conv3d_2c_3x3/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Conv3d_2c_3x3.bn.bias.data.shape == T.shape:
        model._model.Conv3d_2c_3x3.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()


    st = 'Mixed_3b/Branch_0/Conv3d_0a_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_3b.b0.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_3b.b0.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_3b.b0.bn.bias.data.shape == T.shape:
        model._model.Mixed_3b.b0.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_3b/Branch_1/Conv3d_0a_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_3b.b1a.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_3b.b1a.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_3b.b1a.bn.bias.data.shape == T.shape:
        model._model.Mixed_3b.b1a.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_3b/Branch_1/Conv3d_0b_3x3/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_3b.b1b.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_3b.b1b.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_3b.b1b.bn.bias.data.shape == T.shape:
        model._model.Mixed_3b.b1b.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_3b/Branch_2/Conv3d_0a_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_3b.b2a.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_3b.b2a.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_3b.b2a.bn.bias.data.shape == T.shape:
        model._model.Mixed_3b.b2a.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_3b/Branch_2/Conv3d_0b_3x3/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_3b.b2b.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_3b.b2b.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_3b.b2b.bn.bias.data.shape == T.shape:
        model._model.Mixed_3b.b2b.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_3b/Branch_3/Conv3d_0b_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_3b.b3b.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_3b.b3b.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_3b.b3b.bn.bias.data.shape == T.shape:
        model._model.Mixed_3b.b3b.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_3c/Branch_0/Conv3d_0a_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_3c.b0.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_3c.b0.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_3c.b0.bn.bias.data.shape == T.shape:
        model._model.Mixed_3c.b0.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_3c/Branch_1/Conv3d_0a_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_3c.b1a.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_3c.b1a.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_3c.b1a.bn.bias.data.shape == T.shape:
        model._model.Mixed_3c.b1a.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_3c/Branch_1/Conv3d_0b_3x3/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_3c.b1b.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_3c.b1b.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_3c.b1b.bn.bias.data.shape == T.shape:
        model._model.Mixed_3c.b1b.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_3c/Branch_2/Conv3d_0a_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_3c.b2a.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_3c.b2a.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_3c.b2a.bn.bias.data.shape == T.shape:
        model._model.Mixed_3c.b2a.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_3c/Branch_2/Conv3d_0b_3x3/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_3c.b2b.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_3c.b2b.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_3c.b2b.bn.bias.data.shape == T.shape:
        model._model.Mixed_3c.b2b.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_3c/Branch_3/Conv3d_0b_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_3c.b3b.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_3c.b3b.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_3c.b3b.bn.bias.data.shape == T.shape:
        model._model.Mixed_3c.b3b.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4b/Branch_0/Conv3d_0a_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_4b.b0.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_4b.b0.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_4b.b0.bn.bias.data.shape == T.shape:
        model._model.Mixed_4b.b0.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4b/Branch_1/Conv3d_0a_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_4b.b1a.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_4b.b1a.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_4b.b1a.bn.bias.data.shape == T.shape:
        model._model.Mixed_4b.b1a.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4b/Branch_1/Conv3d_0b_3x3/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_4b.b1b.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_4b.b1b.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_4b.b1b.bn.bias.data.shape == T.shape:
        model._model.Mixed_4b.b1b.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4b/Branch_2/Conv3d_0a_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_4b.b2a.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_4b.b2a.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_4b.b2a.bn.bias.data.shape == T.shape:
        model._model.Mixed_4b.b2a.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4b/Branch_2/Conv3d_0b_3x3/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_4b.b2b.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_4b.b2b.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_4b.b2b.bn.bias.data.shape == T.shape:
        model._model.Mixed_4b.b2b.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4b/Branch_3/Conv3d_0b_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_4b.b3b.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_4b.b3b.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_4b.b3b.bn.bias.data.shape == T.shape:
        model._model.Mixed_4b.b3b.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4c/Branch_0/Conv3d_0a_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_4c.b0.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_4c.b0.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_4c.b0.bn.bias.data.shape == T.shape:
        model._model.Mixed_4c.b0.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4c/Branch_1/Conv3d_0a_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_4c.b1a.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_4c.b1a.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_4c.b1a.bn.bias.data.shape == T.shape:
        model._model.Mixed_4c.b1a.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4c/Branch_1/Conv3d_0b_3x3/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_4c.b1b.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_4c.b1b.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_4c.b1b.bn.bias.data.shape == T.shape:
        model._model.Mixed_4c.b1b.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4c/Branch_2/Conv3d_0a_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_4c.b2a.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_4c.b2a.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_4c.b2a.bn.bias.data.shape == T.shape:
        model._model.Mixed_4c.b2a.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4c/Branch_2/Conv3d_0b_3x3/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_4c.b2b.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_4c.b2b.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_4c.b2b.bn.bias.data.shape == T.shape:
        model._model.Mixed_4c.b2b.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4c/Branch_3/Conv3d_0b_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_4c.b3b.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_4c.b3b.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_4c.b3b.bn.bias.data.shape == T.shape:
        model._model.Mixed_4c.b3b.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4d/Branch_0/Conv3d_0a_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_4d.b0.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_4d.b0.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_4d.b0.bn.bias.data.shape == T.shape:
        model._model.Mixed_4d.b0.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4d/Branch_1/Conv3d_0a_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_4d.b1a.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_4d.b1a.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_4d.b1a.bn.bias.data.shape == T.shape:
        model._model.Mixed_4d.b1a.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4d/Branch_1/Conv3d_0b_3x3/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_4d.b1b.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_4d.b1b.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_4d.b1b.bn.bias.data.shape == T.shape:
        model._model.Mixed_4d.b1b.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4d/Branch_2/Conv3d_0a_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_4d.b2a.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_4d.b2a.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_4d.b2a.bn.bias.data.shape == T.shape:
        model._model.Mixed_4d.b2a.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4d/Branch_2/Conv3d_0b_3x3/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_4d.b2b.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_4d.b2b.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_4d.b2b.bn.bias.data.shape == T.shape:
        model._model.Mixed_4d.b2b.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4d/Branch_3/Conv3d_0b_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_4d.b3b.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_4d.b3b.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_4d.b3b.bn.bias.data.shape == T.shape:
        model._model.Mixed_4d.b3b.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4e/Branch_0/Conv3d_0a_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_4e.b0.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_4e.b0.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_4e.b0.bn.bias.data.shape == T.shape:
        model._model.Mixed_4e.b0.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4e/Branch_1/Conv3d_0a_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_4e.b1a.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_4e.b1a.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_4e.b1a.bn.bias.data.shape == T.shape:
        model._model.Mixed_4e.b1a.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4e/Branch_1/Conv3d_0b_3x3/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_4e.b1b.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_4e.b1b.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_4e.b1b.bn.bias.data.shape == T.shape:
        model._model.Mixed_4e.b1b.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4e/Branch_2/Conv3d_0a_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_4e.b2a.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_4e.b2a.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_4e.b2a.bn.bias.data.shape == T.shape:
        model._model.Mixed_4e.b2a.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4e/Branch_2/Conv3d_0b_3x3/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_4e.b2b.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_4e.b2b.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_4e.b2b.bn.bias.data.shape == T.shape:
        model._model.Mixed_4e.b2b.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4e/Branch_3/Conv3d_0b_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_4e.b3b.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_4e.b3b.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_4e.b3b.bn.bias.data.shape == T.shape:
        model._model.Mixed_4e.b3b.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4f/Branch_0/Conv3d_0a_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_4f.b0.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_4f.b0.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_4f.b0.bn.bias.data.shape == T.shape:
        model._model.Mixed_4f.b0.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4f/Branch_1/Conv3d_0a_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_4f.b1a.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_4f.b1a.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_4f.b1a.bn.bias.data.shape == T.shape:
        model._model.Mixed_4f.b1a.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4f/Branch_1/Conv3d_0b_3x3/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_4f.b1b.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_4f.b1b.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_4f.b1b.bn.bias.data.shape == T.shape:
        model._model.Mixed_4f.b1b.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4f/Branch_2/Conv3d_0a_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_4f.b2a.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_4f.b2a.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_4f.b2a.bn.bias.data.shape == T.shape:
        model._model.Mixed_4f.b2a.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4f/Branch_2/Conv3d_0b_3x3/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_4f.b2b.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_4f.b2b.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_4f.b2b.bn.bias.data.shape == T.shape:
        model._model.Mixed_4f.b2b.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4f/Branch_3/Conv3d_0b_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_4f.b3b.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_4f.b3b.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_4f.b3b.bn.bias.data.shape == T.shape:
        model._model.Mixed_4f.b3b.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_5b/Branch_0/Conv3d_0a_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_5b.b0.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_5b.b0.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_5b.b0.bn.bias.data.shape == T.shape:
        model._model.Mixed_5b.b0.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_5b/Branch_1/Conv3d_0a_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_5b.b1a.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_5b.b1a.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_5b.b1a.bn.bias.data.shape == T.shape:
        model._model.Mixed_5b.b1a.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_5b/Branch_1/Conv3d_0b_3x3/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_5b.b1b.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_5b.b1b.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_5b.b1b.bn.bias.data.shape == T.shape:
        model._model.Mixed_5b.b1b.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_5b/Branch_2/Conv3d_0a_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_5b.b2a.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_5b.b2a.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_5b.b2a.bn.bias.data.shape == T.shape:
        model._model.Mixed_5b.b2a.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_5b/Branch_2/Conv3d_0a_3x3/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_5b.b2b.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_5b.b2b.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_5b.b2b.bn.bias.data.shape == T.shape:
        model._model.Mixed_5b.b2b.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_5b/Branch_3/Conv3d_0b_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_5b.b3b.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_5b.b3b.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_5b.b3b.bn.bias.data.shape == T.shape:
        model._model.Mixed_5b.b3b.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()



    st = 'Mixed_5c/Branch_0/Conv3d_0a_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_5c.b0.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_5c.b0.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_5c.b0.bn.bias.data.shape == T.shape:
        model._model.Mixed_5c.b0.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_5c/Branch_1/Conv3d_0a_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_5c.b1a.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_5c.b1a.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_5c.b1a.bn.bias.data.shape == T.shape:
        model._model.Mixed_5c.b1a.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_5c/Branch_1/Conv3d_0b_3x3/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_5c.b1b.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_5c.b1b.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_5c.b1b.bn.bias.data.shape == T.shape:
        model._model.Mixed_5c.b1b.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_5c/Branch_2/Conv3d_0a_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_5c.b2a.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_5c.b2a.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_5c.b2a.bn.bias.data.shape == T.shape:
        model._model.Mixed_5c.b2a.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_5c/Branch_2/Conv3d_0b_3x3/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_5c.b2b.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_5c.b2b.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_5c.b2b.bn.bias.data.shape == T.shape:
        model._model.Mixed_5c.b2b.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_5c/Branch_3/Conv3d_0b_1x1/conv_3d/w'; T = convert(state_dict, st)
    if model._model.Mixed_5c.b3b.conv3d.weight.data.shape == T.shape:
        model._model.Mixed_5c.b3b.conv3d.weight.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    st = 'Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/beta'; T = convert(state_dict, st)
    if model._model.Mixed_5c.b3b.bn.bias.data.shape == T.shape:
        model._model.Mixed_5c.b3b.bn.bias.data = T
    else:
        print("shape mismatch"); pdb.set_trace();
        quit()

    return model





# filename='video_validation_0000051.mp4'
# featfile = 'video_validation_0000051.npy'
# feats = np.load(featfile)
# print(feats.shape)
# V = torchvision.io.read_video(filename, 0, 100)[0]
# print(V.shape)


# # getting the first few frames of video file to test feature similarity
# V = V[0:16, :, :, :]

# V = V.unsqueeze(0)
# V = V.permute(0,4,1,2,3) # permute should be preferred over reshape

# #print(torch.max(V))
# V = V/255.0


# a2 = torch.zeros(1,3,16,224,224)
# a = V#torch.rand(1,3,16,180,320)

# # processing the video data
# for i in range(3):
#   crop = torchvision.transforms.CenterCrop(224)
#   res = torchvision.transforms.Resize((256, 455))
#   b = a[:,i,:,:,:]
#   print(b.shape)
#   c = res(b)
#   d = crop(c)
#   print(d.shape)
#   a2[:,i,:,:,:] = d


# print(a2.shape)

# # loading up the video
# V=a2


# V2 = model(V)
# feats2 = V2[0,:,0,0,0]
# print(feats2.shape)
# print("done")
