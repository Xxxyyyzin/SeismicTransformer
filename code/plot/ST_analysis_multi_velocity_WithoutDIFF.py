"""
This script is :Transformer模型下绘制多个速度模型比较，不含差分图
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from analysis_SeisTrans import load_model_SeisTrans, load_testdataset_SeisTrans
from criterion_single_function import EvaluationValueALL_single

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

model100thous, c_dict100thous = load_model_SeisTrans("fault_SeismicTrans_Adam_l1_mean_loss_gain", rootdir="server/",optimizer="Adam",nth_model=4,verbose=False)
d = load_testdataset_SeisTrans("fault_2ms_r_generalisation.bin",rootdir="../generate_data_main/data/", N_EXAMPLES=30, c_dict=c_dict100thous, verbose=False)   #N_EXAMPLES 表示数据集下有多少个数据

# Get batches of test data
irange = np.arange(30)   #加载测试集的数量

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
    model100thous.eval()
    outputs100thous = model100thous(*inputs)

    inputs_array = inputs[0].detach().cpu().numpy().copy()# detach returns a new tensor, detached from the current graph
    source_array = inputs[1].detach().cpu().numpy().copy()
    labels_array = labels[0].detach().cpu().numpy().copy()
    outputs_array100thous = outputs100thous.numpy()

# un-normalise velocity
inputs_array = c_dict100thous["VELOCITY_SIGMA"]*inputs_array + c_dict100thous["VELOCITY_MU"]
# ibs = np.arange(0, 3* 4, 3) + 0  # as velocity model goes in threes 等差数列确定画哪些图
#[0 1 2 3 4 5 6 7]
ibs=np.array([2,11,25])    #自定义画第几张图

f = plt.figure(figsize=0.8 * np.array([14.5, 12.5]))

for i in range(0, 3):
    ib = ibs[i]

    # velocity
    ncol = 12
    nrow = 4

    ax = f.add_axes([1 / ncol, (nrow - (i+1)) / nrow, 2 / ncol, 1 / nrow])  # xmin, ymin, dx, and dy
    im0 = plt.imshow(inputs_array[ib, 0, :, :].T, vmin=CLIM[0], vmax=CLIM[1])  # 速度模型
    plt.scatter(NX * source_array[ib, 0, :, :], NX * source_array[0, 1, :, :], c="white")  # 加上source


    if i >=2:
        plt.xticks(np.arange(0, NX, 40), (DX * np.arange(0, NX, 40)).astype(np.int))   #最后两幅，加上横轴坐标
        plt.xlabel("Distance (m)")     #最后两幅，加上横轴单位
    else:
        plt.xticks([])   #其他位置不加

    plt.yticks(np.arange(0, NZ, 40)[::-1], (DZ * np.arange(0, NZ, 40)[::-1]).astype(np.int))   #最左边的加上纵坐标
    plt.ylabel("Depth (m)")     #最左边的加上纵轴单位
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
    if i >=2:
        plt.xticks(np.arange(0, NREC, 8), (DX * DELTARECi * np.arange(0, NREC, 8)).astype(np.int))   #最后两幅，加上横轴坐标
        plt.xlabel("Receiver offset (m)")    #最后两幅，加上横轴名称
        plt.xticks(rotation=50)

    # gather (NN5)
    ax = f.add_axes([ 5.2 / ncol, (nrow - (i+1)) / nrow, 0.9 / ncol, 1 / nrow])# xmin, ymin, dx, and dy
    im1 = plt.imshow((t_gain * outputs_array100thous)[ib, 0, :, :].T,aspect=0.2, cmap="gray_r", vmin=-VLIM, vmax=VLIM)
    plt.xticks([])
    plt.yticks([])
    plt.ylim(400, 0)
    if i==0:
        plt.title("SeismicTrans\n100000th")

    # # gather (diff)
    # ax = f.add_axes([ 6.4 / ncol, (nrow - (i+1)) / nrow, 0.9 / ncol, 1 / nrow])  # xmin, ymin, dx, and dy
    # plt.imshow((t_gain * (labels_array - outputs_array100thous))[ib, 0, :, :].T,aspect=0.2, cmap="gray_r", vmin=-VLIM, vmax=VLIM)
    # plt.xticks([])
    # plt.yticks([])
    # plt.ylim(400, 0)
    # if i == 0:
    #     plt.title("Difference")


    if i == 0:      #在第一幅最右边加上色彩条
        ax = f.add_axes([6.2 / ncol, 3.1 / nrow, 0.15 / ncol, 0.8 / nrow])
        cb = f.colorbar(im0, cax=ax)
        cb.ax.set_ylabel('Velocity ($\mathrm{ms}^{-1}$)')

        ax = f.add_axes([6.2 / ncol, 2.1 / nrow, 0.15 / ncol, 0.8 / nrow])
        cb = f.colorbar(im1, cax=ax, aspect=0.01)
        cb.ax.set_ylabel('Pressure (arb)')

    # 每一幅图的评价值
    print(i)
    value_trans_i=EvaluationValueALL_single(labels_array[ib,0,:,:],outputs_array100thous[ib,0,:,:],irange)
    print(value_trans_i)

plt.savefig("../report_plots/multi_veloctity_1111.pdf", bbox_inches='tight', pad_inches=0.01, dpi=400)
plt.show()