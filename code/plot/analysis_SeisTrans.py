"""
This script is :Transformer模型下绘制的外部函数
"""
import os
import pickle
import torch
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from datasets import SeismicBinaryDBDataset

import sys
sys.path.insert(0, '../shared_modules/')
from helper import DictToObj
sys.path.insert(0, '../Transformer/')      #transformer


def load_model_SeisTrans(MODEL_LOAD_RUN, rootdir="server/",optimizer="Adam",nth_model=0,verbose=False):
    """load a model and its constants object from rootdir.
    MODEL_LOAD_RUN can be of form 'model' or 'model/model_i.torch' """
    
    rootdir = rootdir.rstrip("/")+"/"
    
    # parse MODEL_RUN and MODEL_DIR from path
    MODEL_LOAD_RUN = MODEL_LOAD_RUN.rstrip("/").split("/")
    MODEL_RUN = MODEL_LOAD_RUN[0]
    MODEL_DIR = rootdir+"models/%s/%s/"%(optimizer,MODEL_RUN)
    
    # parse specific model file
    if len(MODEL_LOAD_RUN) == 2:# if model file specified, load that model
        MODEL = MODEL_LOAD_RUN[1]
    else:# else load the final model in directory
        model_files = sorted(os.listdir(MODEL_DIR))
        MODEL = model_files[nth_model]
    
    # load constants dicionary
    CONSTANTS = "constants_%s.pickle"%(MODEL_RUN)
    SUMMARY_DIR = rootdir+"summaries/%s/%s/"%(optimizer,MODEL_RUN)
    if verbose: print("Loading constants: %s"%(SUMMARY_DIR+CONSTANTS))
    c_dict = pickle.load(open(SUMMARY_DIR+CONSTANTS, "rb"))
    c = DictToObj(**c_dict, copy=True)# convert to object
    if verbose: print(c)
    
    # restore a model using model file and constants file
    print("Loading model from: %s"%(MODEL_DIR+MODEL))
    model = c.MODEL()# initialise model ! uses current code (pickle only saves name of class)
    cp = torch.load(MODEL_DIR+MODEL,map_location=torch.device('cpu'))# remap tensors from gpu to cpu if needed
    model.load_state_dict(cp['model_state_dict'])
    model.eval()
    if verbose: print(model)
    
    #optimizer = torch.optim.Adam(model.parameters(), lr=c.LRATE)
    #optimizer.load_state_dict(cp['optimizer_state_dict'])
    
    return model, c_dict


def load_testdataset_SeisTrans(DATA_PATH,
                     N_EXAMPLES,
                     c_dict,
                     rootdir="../generate_data/data/",
                     verbose=False):
    """load a test dataset using constants from c_dict"""
    
    rootdir = rootdir.rstrip("/")+"/"
    
    if verbose: print("Loading testdataset from %s%s"%(rootdir,DATA_PATH))
    
    # build c_eval object
    c_eval = DictToObj(**c_dict)
    
    # override fields
    c_eval.N_EXAMPLES = N_EXAMPLES
    c_eval.DATA_PATH = rootdir+DATA_PATH
    
    # delete any output dirs for safety
    for name in ["OUT_DIR", "MODEL_OUT_DIR", "SUMMARY_OUT_DIR"]:# clear out dir paths
        if hasattr(c_eval, name): delattr(c_eval, name)
    if verbose: print(c_eval)
    
    # load a test dataset using this evaluation constants object
    # use SeismicDataset for simplicity
    testdataset = SeismicBinaryDBDataset(c_eval,
                         irange=np.arange(c_eval.N_EXAMPLES),# load all examples
                         verbose=True)
    
    return testdataset


def plot_result_SeisTrans(inputs_array, outputs_array, labels_array, sample_batch=None, ib=0, isource=0,
                aspect=0.2, T_GAIN=2.5, vmin=-1, vmax=1, gmin=-1, gmax=1,title="SeismicTrans\n100000th"):
    "Plot a network prediction, compare to ground truth and input"
    f = plt.figure(figsize=(12,5))
    
    # define gain profile for display
    t_gain = np.arange(outputs_array.shape[-1], dtype=np.float32)**T_GAIN
    t_gain = t_gain/np.median(t_gain)
    t_gain = t_gain.reshape((1,1,1,outputs_array.shape[-1]))# along NSTEPS

    #速度模型
    plt.subplot2grid((1, 5), (0, 0), colspan=2)
    plt.imshow(inputs_array[ib,0,:,:].T, vmin=vmin, vmax=vmax)
    plt.xlabel("Distance (m)")
    plt.ylabel("Depth (m)")
    plt.colorbar()

    # 真实地震记录
    plt.subplot2grid((1, 5), (0, 2), colspan=1)
    plt.imshow((t_gain * labels_array)[ib, isource, :, :].T, aspect=aspect, cmap="Greys", vmin=gmin, vmax=gmax)
    plt.title("Ground\ntruth")
    plt.xlabel("Receiver offset (m)")
    plt.xticks(rotation=50)

    # plt.title("%f, %f"%(np.min(outputs_array),np.max(outputs_array)))
    # 网络预测得到的地震记录
    plt.subplot2grid((1, 5), (0, 3), colspan=1)
    plt.imshow((t_gain * outputs_array)[ib, isource, :, :].T, aspect=aspect, cmap="Greys", vmin=gmin, vmax=gmax)
    plt.title(title)

    #差分图
    plt.subplot2grid((1, 5), (0, 4), colspan=1)
    plt.imshow((t_gain * (labels_array - outputs_array))[ib,isource,:,:].T,aspect=aspect, cmap="Greys", vmin=gmin, vmax=gmax)
    plt.title("Difference")
    plt.colorbar()

    if type(sample_batch)!=type(None):
        plt.title("%s"%(sample_batch["inputs"][1].detach().cpu().numpy().copy()[ib,:,0,0]))# label with source position
    
    return f


def plot_result_withoutDIFF_SeisTrans(inputs_array, outputs_array, labels_array, sample_batch=None, ib=0, isource=0,
                aspect=0.2, T_GAIN=2.5, vmin=-1, vmax=1, gmin=-1, gmax=1):
    "Plot a network prediction, compare to ground truth and input"
    f = plt.figure(figsize=(12, 5))

    # define gain profile for display
    t_gain = np.arange(outputs_array.shape[-1], dtype=np.float32) ** T_GAIN
    t_gain = t_gain / np.median(t_gain)
    t_gain = t_gain.reshape((1, 1, 1, outputs_array.shape[-1]))  # along NSTEPS

    plt.subplot2grid((1, 4), (0, 0), colspan=2)
    plt.imshow(inputs_array[ib, 0, :, :].T, vmin=vmin, vmax=vmax)
    plt.xlabel("Distance (m)")
    plt.ylabel("Depth (m)")
    plt.colorbar()

    plt.subplot2grid((1, 4), (0, 2), colspan=1)
    plt.imshow((t_gain * labels_array)[ib, isource, :, :].T, aspect=aspect, cmap="Greys", vmin=gmin, vmax=gmax)
    plt.title("Ground\ntruth")
    plt.xlabel("Receiver offset (m)")
    plt.xticks(rotation=50)

    # plt.title("%f, %f"%(np.min(outputs_array),np.max(outputs_array)))

    plt.subplot2grid((1, 4), (0, 3), colspan=1)
    plt.imshow((t_gain * outputs_array)[ib, isource, :, :].T, aspect=aspect, cmap="Greys", vmin=gmin, vmax=gmax)
    plt.title("SeismicTrans\n100000th")
    plt.colorbar()

    if type(sample_batch) != type(None):
        plt.title(
            "%s" % (sample_batch["inputs"][1].detach().cpu().numpy().copy()[ib, :, 0, 0]))  # label with source position

    return f


def citerion_l1(true_gather,predict_gather):
    l1=np.sum(np.abs(true_gather-predict_gather))
    return l1

def citerion_lendless(true_gather,predict_gather):
    lendless = np.max(np.abs(true_gather-predict_gather))
    return lendless