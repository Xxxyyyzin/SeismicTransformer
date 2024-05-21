"""
This script is: Draw prediction graphs for different iterations under the AutoEncoder model
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from analysis_Conv import load_model_Conv, load_testdataset_Conv
from criterion_single_function import EvaluationValueALL_single


# Global constants
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
letters = ['a','b','c','d','g']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

# Load model and dataset
model20thous, c_dict20thous = load_model_Conv("fault_AE_r_Adam_l1_mean_loss_gain(t_gain=2.5)", rootdir="server/",optimizer="Adam",nth_model=9,verbose=False) #nth_model 导入网络在文件夹下的索引
model40thous, c_dict40thous = load_model_Conv("fault_AE_r_Adam_l1_mean_loss_gain(t_gain=2.5)", rootdir="server/",optimizer="Adam",nth_model=19,verbose=False)
model60thous, c_dict60thous = load_model_Conv("fault_AE_r_Adam_l1_mean_loss_gain(t_gain=2.5)", rootdir="server/",optimizer="Adam",nth_model=29,verbose=False)
model80thous, c_dict80thous = load_model_Conv("fault_AE_r_Adam_l1_mean_loss_gain(t_gain=2.5)", rootdir="server/",optimizer="Adam",nth_model=39,verbose=False)
model100thous, c_dict100thous = load_model_Conv("fault_AE_r_Adam_l1_mean_loss_gain(t_gain=2.5)", rootdir="server/",optimizer="Adam",nth_model=49,verbose=False)
d = load_testdataset_Conv("fault_2ms_r_generalisation.bin",rootdir="../generate_data_main/data/", N_EXAMPLES=30, c_dict=c_dict20thous, verbose=False)   #N_EXAMPLES 表示数据集下有多少个数据

# Get batches of test data
irange = np.arange(30)   # Number of loaded test sets

d.open_file_reader()
samples = [d[i] for i in irange]
# print(samples)
# d.close_file_reader()
inputs = [torch.cat([sample["inputs"][i].unsqueeze(0) for sample in samples]) for i in range(len(samples[0]["inputs"]))]
labels = [torch.cat([sample["labels"][i].unsqueeze(0) for sample in samples]) for i in range(len(samples[0]["labels"]))]
for i in inputs: print(i.shape)
for i in labels: print(i.shape)

# Inference
with torch.no_grad():# faster inference without tracking
    model20thous.eval()
    outputs20thous = model20thous(*inputs)

    model40thous.eval()
    outputs40thous = model40thous(*inputs)

    model60thous.eval()
    outputs60thous = model60thous(*inputs)

    model80thous.eval()
    outputs80thous = model80thous(*inputs)

    model100thous.eval()
    outputs100thous = model100thous(*inputs)

    inputs_array = inputs[0].detach().cpu().numpy().copy()# detach returns a new tensor, detached from the current graph
    source_array = inputs[1].detach().cpu().numpy().copy()
    labels_array = labels[0].detach().cpu().numpy().copy()
    outputs_array20thous = outputs20thous[0].detach().cpu().numpy().copy()
    outputs_array40thous = outputs40thous[0].detach().cpu().numpy().copy()
    outputs_array60thous = outputs60thous[0].detach().cpu().numpy().copy()
    outputs_array80thous = outputs80thous[0].detach().cpu().numpy().copy()
    outputs_array100thous = outputs100thous[0].detach().cpu().numpy().copy()

# un-normalise velocity
inputs_array = c_dict20thous["VELOCITY_SIGMA"]*inputs_array + c_dict20thous["VELOCITY_MU"]
# ibs = np.arange(0, 3* 4, 3) + 0  
#[0 1 2 3 4 5 6 7]
ibs=np.array([2,11,25,8])    # Customize which image to draw

f = plt.figure(figsize=0.8 * np.array([14.5, 12.5]))

for i in range(0, 4):
    ib = ibs[i]

    # velocity
    ncol = 12
    nrow = 4

    ax = f.add_axes([1 / ncol, (nrow - (i+1)) / nrow, 2 / ncol, 1 / nrow])  # xmin, ymin, dx, and dy
    im0 = plt.imshow(inputs_array[ib, 0, :, :].T, vmin=CLIM[0], vmax=CLIM[1])  
    plt.scatter(NX * source_array[ib, 0, :, :], NX * source_array[0, 1, :, :], c="white")  # 加上source

    if i >=3:
        plt.xticks(np.arange(0, NX, 40), (DX * np.arange(0, NX, 40)).astype(np.int))   
        plt.xlabel("Distance (m)")    
    else:
        plt.xticks([])   

    plt.yticks(np.arange(0, NZ, 40)[::-1], (DZ * np.arange(0, NZ, 40)[::-1]).astype(np.int))   
    plt.ylabel("Depth (m)")    
    plt.xlim(0, NX - 1)
    plt.ylim(NZ - 1, 0)
    plt.title("(%s)" % (letters[i]))

    # gather (Ground truth)
    ax = f.add_axes([ 4/ ncol, (nrow - (i+1) ) / nrow, 0.9 / ncol, 1 / nrow])  # xmin, ymin, dx, and dy
    plt.imshow((t_gain * labels_array)[ib, 0, :, :].T, aspect=0.2, cmap="gray_r", vmin=-VLIM, vmax=VLIM)
    plt.yticks(np.arange(0, NSTEPS, 50)[::-1], (["%.1f" % (val) for val in DT * np.arange(0, NSTEPS, 50)[::-1]]))
    plt.xticks([])
    plt.ylim(400, 0)
    if i == 0:
        plt.title("Ground\ntruth")
    if i >=3:
        plt.xticks(np.arange(0, NREC, 8), (DX * DELTARECi * np.arange(0, NREC, 8)).astype(np.int))   
        plt.xlabel("Receiver offset (m)")   
        plt.xticks(rotation=50)

     # gather (NN1)
    ax = f.add_axes([ 5.2/ ncol, (nrow -(i+1) ) / nrow, 0.9 / ncol, 1 / nrow])# xmin, ymin, dx, and dy
    im1 = plt.imshow((t_gain * outputs_array20thous)[ib, 0, :, :].T,aspect=0.2, cmap="gray_r", vmin=-VLIM, vmax=VLIM)
    plt.xticks([])
    plt.yticks([])
    plt.ylim(400, 0)
    if i==0:
        plt.title("AE\n20000th")

    # gather (NN2)
    ax = f.add_axes([ 6.4 / ncol, (nrow - (i+1)) / nrow, 0.9 / ncol, 1 / nrow])# xmin, ymin, dx, and dy
    im1 = plt.imshow((t_gain * outputs_array40thous)[ib, 0, :, :].T,aspect=0.2, cmap="gray_r", vmin=-VLIM, vmax=VLIM)
    plt.xticks([])
    plt.yticks([])
    plt.ylim(400, 0)
    if i==0:
        plt.title("AE\n40000th")

    # gather (NN3)
    ax = f.add_axes([ 7.6/ ncol, (nrow -(i+1)) / nrow, 0.9 / ncol, 1 / nrow])# xmin, ymin, dx, and dy
    im1 = plt.imshow((t_gain * outputs_array60thous)[ib, 0, :, :].T,aspect=0.2, cmap="gray_r", vmin=-VLIM, vmax=VLIM)
    plt.xticks([])
    plt.yticks([])
    plt.ylim(400, 0)
    if i==0:
        plt.title("AE\n60000th")

    # gather (NN4)
    ax = f.add_axes([ 8.8 / ncol, (nrow -(i+1)) / nrow, 0.9 / ncol, 1 / nrow])# xmin, ymin, dx, and dy
    im1 = plt.imshow((t_gain * outputs_array80thous)[ib, 0, :, :].T,aspect=0.2, cmap="gray_r", vmin=-VLIM, vmax=VLIM)
    plt.xticks([])
    plt.yticks([])
    plt.ylim(400, 0)
    if i==0:
        plt.title("AE\n80000th")

    # gather (NN5)
    ax = f.add_axes([ 10.0 / ncol, (nrow - (i+1)) / nrow, 0.9 / ncol, 1 / nrow])# xmin, ymin, dx, and dy
    im1 = plt.imshow((t_gain * outputs_array100thous)[ib, 0, :, :].T,aspect=0.2, cmap="gray_r", vmin=-VLIM, vmax=VLIM)
    plt.xticks([])
    plt.yticks([])
    plt.ylim(400, 0)
    if i==0:
        plt.title("AE\n100000th")


    if i == 0:     
        ax = f.add_axes([11 / ncol, 3.1 / nrow, 0.15 / ncol, 0.8 / nrow])
        cb = f.colorbar(im0, cax=ax)
        cb.ax.set_ylabel('Velocity ($\mathrm{ms}^{-1}$)')

        ax = f.add_axes([11 / ncol, 2.1 / nrow, 0.15 / ncol, 0.8 / nrow])
        cb = f.colorbar(im1, cax=ax, aspect=0.01)
        cb.ax.set_ylabel('Pressure (arb)')

  
    print("Conv")
    print(i)
    print("20thousands:")
    value_conv_20 = EvaluationValueALL_single(labels_array[ib,0,:,:],outputs_array20thous[ib,0,:,:],irange)
    print(value_conv_20)

    print("40thousands:")
    value_conv_40 = EvaluationValueALL_single(labels_array[ib,0,:,:],outputs_array40thous[ib,0,:,:],irange)
    print(value_conv_40)

    print("60thousands:")
    value_conv_60 = EvaluationValueALL_single(labels_array[ib,0,:,:],outputs_array60thous[ib,0,:,:],irange)
    print(value_conv_60)

    print("80thousands:")
    value_conv_80 = EvaluationValueALL_single(labels_array[ib,0,:,:],outputs_array80thous[ib,0,:,:],irange)
    print(value_conv_80)

    print("100thousands:")
    value_conv_100 = EvaluationValueALL_single(labels_array[ib,0,:,:],outputs_array100thous[ib,0,:,:],irange)
    print(value_conv_100)

plt.savefig("../report_plots/fig1_CONV.pdf", bbox_inches='tight', pad_inches=0.01, dpi=400)
plt.show()

