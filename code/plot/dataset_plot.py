# -*- encoding:utf-8 -*-
"""
@Author：YanJin Xiang
@Email：yanzinxyz@gmail.com
@Date：2021/12/19  下午9:08
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from analysis_SeisTrans import load_model_SeisTrans, load_testdataset_SeisTrans

DT = 0.002
T_GAIN = 2.5
DX,DZ = 5., 5.
NX,NZ = 128, 128
NREC,NSTEPS = 32, 512
DELTARECi = 3


# define gain profile for display
t_gain = np.arange(NSTEPS, dtype=np.float32)**T_GAIN
t_gain = t_gain/np.median(t_gain)
t_gain = t_gain.reshape((1,1,1,NSTEPS))# along NSTEPS
VLIM = 0.6
CLIM = (1500,3600)
letters = ['a','b','c']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


model100thous, c_dict100thous = load_model_SeisTrans("fault_SeismicTrans_Adam_l1_mean_loss_gain", rootdir="server/",optimizer="Adam",nth_model=0,verbose=False)
d = load_testdataset_SeisTrans("fault_2ms_r.bin",rootdir="../generate_data_main/data/", N_EXAMPLES=10, c_dict=c_dict100thous, verbose=False)   #N_EXAMPLES 表示数据集下有多少个数据

# Get batches of test data
irange = np.arange(10)

d.open_file_reader()
samples = [d[i] for i in irange]
d.close_file_reader()
inputs = [torch.cat([sample["inputs"][i].unsqueeze(0) for sample in samples]) for i in range(len(samples[0]["inputs"]))]
labels = [torch.cat([sample["labels"][i].unsqueeze(0) for sample in samples]) for i in range(len(samples[0]["labels"]))]
for i in inputs: print(i.shape)
for i in labels: print(i.shape)

# Inference
with torch.no_grad():# faster inference without tracking
    # model100thous.eval()
    # outputs100thous = model100thous(*inputs)

    inputs_array = inputs[0].detach().cpu().numpy().copy()# detach returns a new tensor, detached from the current graph
    source_array = inputs[1].detach().cpu().numpy().copy()
    # outputs_array = outputs[0].detach().cpu().numpy().copy()
    labels_array = labels[0].detach().cpu().numpy().copy()
    # outputs_array100thous=outputs100thous.numpy()

# un-normalise velocity
inputs_array = c_dict100thous["VELOCITY_SIGMA"]*inputs_array + c_dict100thous["VELOCITY_MU"]

f = plt.figure(figsize=(15, 5))
for i in range(3):

    # velocity
    plt.subplot2grid((3, 30), (i, 0), colspan=4)
    im0 = plt.imshow(inputs_array[3* i, 0, :, :].T, vmin=CLIM[0], vmax=CLIM[1])
    plt.scatter(NX * source_array[3* i:3 * i +3, 0, 0, 0], NX * source_array[3 * i:3 * i +3, 1, 0, 0], c="white")
    plt.gca().set_anchor('C')  # centre plot
    if i == 2:
        plt.xticks(np.arange(0, NX, 40), (DX * np.arange(0, NX, 40)).astype(np.int))
        plt.xlabel("Distance (m)")
    else:
        plt.xticks([])
    plt.yticks(np.arange(0, NZ, 40)[::-1], (DZ * np.arange(0, NZ, 40)[::-1]).astype(np.int))
    plt.xlim(0, NX - 1)
    plt.ylim(NZ - 1, 0)
    plt.ylabel("Depth (m)")
    plt.title("(%s)" % (letters[i]))

    print("第%s个速度模型:" % (i + 1))

    for source_i in range(3):

        # ground truth
        plt.subplot2grid((3, 28), (i, 4 + 2 * source_i), colspan=2)
        im1=plt.imshow((t_gain * labels_array)[3 * i + source_i, 0, :, :].T,aspect=0.12, cmap="gray_r", vmin=-VLIM, vmax=VLIM)
        if i == 2 and source_i == 0:
            plt.xticks(np.arange(0, NREC, 8), (DX * DELTARECi * np.arange(0, NREC, 8)).astype(np.int))
            plt.xlabel("Receiver offset (m)")
            plt.xticks(rotation=50)
        else:
            plt.xticks([])
        plt.yticks([])
        plt.ylim(400, 0)
        plt.gca().set_anchor('E')  # centre plot
        if source_i == 0 and i == 0:
            plt.title("Ground\ntruth")


    if i % 2 == 1:
        ax = f.add_axes([0.41, 0.68, 0.01, 0.20])
        cb = f.colorbar(im0, cax=ax)
        cb.ax.set_ylabel('Velocity ($\mathrm{ms}^{-1}$)')

        ax = f.add_axes([0.41, 0.405, 0.01, 0.20])
        cb = f.colorbar(im1, cax=ax, aspect=0.01)
        cb.ax.set_ylabel('Pressure (arb)')

plt.savefig("plot.pdf", bbox_inches='tight', pad_inches=0.01, dpi=300)
plt.savefig("plot.png", bbox_inches='tight', pad_inches=0.01, dpi=300)
plt.show()