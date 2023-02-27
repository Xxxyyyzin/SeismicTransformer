import numpy as np
import matplotlib.pyplot as plt
import torch
from analysis import load_model, load_testdataset
import multiprocessing as mp

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
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

# Load all training velocity models
vels = np.stack([np.load("../marmousi/generate_marmousi/velocity/marmousi/velocity_%.8i.npy"%(i)) for i in range(1000)],axis=0)
print(vels.shape)

# Load model and dataset (marmousi)
model100thous, c_dict100thous = load_model("fault_SeismicTrans_Adam_l1_mean_loss_gain", rootdir="server/",optimizer="Adam",nth_model=0,verbose=False)
d = load_testdataset("marmousi_2ms.bin", rootdir="../marmousi/data/", N_EXAMPLES=3*1000, c_dict=c_dict100thous, verbose=False)

# Get batches of  marmousi test data
irange = np.array([86,35,75,57])*3   #四个marmousi模型速度模型的索引

d.open_file_reader()
samples = [d[i] for i in irange]
d.close_file_reader()
inputs = [torch.cat([sample["inputs"][i].unsqueeze(0) for sample in samples]) for i in range(len(samples[0]["inputs"]))]
labels = [torch.cat([sample["labels"][i].unsqueeze(0) for sample in samples]) for i in range(len(samples[0]["labels"]))]
for i in inputs: print(i.shape)
for i in labels: print(i.shape)

# Inference
with torch.no_grad():# faster inference without tracking
    model100thous.eval()
    outputs = model100thous(*inputs)

    inputs_array = inputs[0].detach().cpu().numpy().copy()# detach returns a new tensor, detached from the current graph
    source_array = inputs[1].detach().cpu().numpy().copy()
    # outputs_array = outputs[0].detach().cpu().numpy().copy()
    labels_array = labels[0].detach().cpu().numpy().copy()
    outputs_array100thous=outputs.numpy()

# Load model and dataset
model100thous, c_dict100thous = load_model("fault_SeismicTrans_Adam_l1_mean_loss_gain", rootdir="server/",optimizer="Adam",nth_model=0,verbose=False)
d = load_testdataset("fault_2ms_r_generalisation.bin",rootdir="../generate_data_main/data/", N_EXAMPLES=30, c_dict=c_dict100thous, verbose=False)   #N_EXAMPLES 表示数据集下有多少个数据

# Get batches of test data
irange = np.array([8,1,5,2]) * 3   #四个断层速度模型的索引

d.open_file_reader()
samples = [d[i] for i in irange]
d.close_file_reader()
inputs = [torch.cat([sample["inputs"][i].unsqueeze(0) for sample in samples]) for i in range(len(samples[0]["inputs"]))]
labels = [torch.cat([sample["labels"][i].unsqueeze(0) for sample in samples]) for i in range(len(samples[0]["labels"]))]
for i in inputs: print(i.shape)
for i in inputs: print(i.shape)
for i in labels: print(i.shape)

# Inference
with torch.no_grad():  # faster inference without tracking
    model100thous.eval()
    outputs= model100thous(*inputs)
    outputs=outputs.numpy()

    
    #将marmousi模型和断层模型拼接到一起[0,1,2,3,4,5,6,7] 索引0-3为marmousi模型；4-7为断层模型
    inputs_array = np.concatenate([inputs_array, inputs[0].detach().cpu().numpy().copy()])  # detach returns a new tensor, detached from the current graph
    source_array = np.concatenate([source_array, inputs[1].detach().cpu().numpy().copy()])
    # outputs_array = np.concatenate([outputs_array100thous, outputs[0].detach().cpu().numpy().copy()])
    labels_array = np.concatenate([labels_array, labels[0].detach().cpu().numpy().copy()])
    outputs_array = np.concatenate([outputs_array100thous, outputs])   #outputs_array.shape:[8,1,32,512]

# un-normalise velocity
inputs_array = c_dict100thous["VELOCITY_SIGMA"] * inputs_array + c_dict100thous["VELOCITY_MU"]   #不能去掉


# # find nearest neighbours
# split = np.array_split(vels, 8)
# neighbours = []
# for ib in np.arange(inputs_array.shape[0]):
#     vel = inputs_array[ib:ib+1,0]
#     def loss_func(i): return np.mean(np.abs(split[i]-vel), axis=(1,2))
#     with mp.Pool(processes=8) as pool: loss = np.concatenate(pool.map(loss_func, np.arange(8)))
#     print(len(loss))
#     a = np.argsort(loss)
#     neighbours.append(a[0])
# print(neighbours)

ibs = [4,5,6,7,0,1,2,3]    #图像如何排列索引0-3为marmousi模型；4-7为断层模型
titles = ["(a) (test set)", "(b) (test set)", "(c) (test set)", "(d) (test set)", "(e) (Marmousi)", "(f) (Marmousi)", "(g) (Marmousi)", "(h) (Marmousi)"]
f = plt.figure(figsize=0.8 * np.array([14.5, 12.5]))
for i in range(0, 8):
    ib = ibs[i]

    # velocity
    ncol = 12
    nrow = 4
    ax = f.add_axes( [(6 * (i % 2) + 0) / ncol, (nrow - 1 - i // 2) / nrow, 2 / ncol, 1 / nrow])  # xmin, ymin, dx, and dy

    im0 = plt.imshow(inputs_array[ib, 0, :, :].T, vmin=CLIM[0], vmax=CLIM[1])
    plt.scatter(NX * source_array[ib, 0, :, :], NX * source_array[ib, 1, :, :], c="white")
    if i // 2 == 3:
        plt.xticks(np.arange(0, NX, 40), (DX * np.arange(0, NX, 40)).astype(np.int))
        plt.xlabel("Distance (m)")
    else:
        plt.xticks([])
    if i % 2 == 0:
        plt.yticks(np.arange(0, NZ, 40)[::-1], (DZ * np.arange(0, NZ, 40)[::-1]).astype(np.int))
        plt.ylabel("Depth (m)")
    else:
        plt.yticks([])
    plt.xlim(0, NX - 1)
    plt.ylim(NZ - 1, 0)
    plt.title("(%s)" % (titles[i]))

    # gather (NN)
    ax = f.add_axes(
        [(6 * (i % 2) + 2.7) / ncol, (nrow - 1 - i // 2) / nrow, 0.9 / ncol, 1 / nrow])  # xmin, ymin, dx, and dy

    im1 = plt.imshow((t_gain * outputs_array)[ib, 0, :, :].T,
                     aspect=0.2, cmap="gray_r", vmin=-VLIM, vmax=VLIM)
    if i // 2 == 3:
        plt.xticks(np.arange(0, NREC, 8), (DX * DELTARECi * np.arange(0, NREC, 8)).astype(np.int))
        plt.xlabel("Receiver offset (m)")
        plt.xticks(rotation=50)
    else:
        plt.xticks([])
    plt.yticks(np.arange(0, NSTEPS, 50)[::-1], (["%.1f" % (val) for val in DT * np.arange(0, NSTEPS, 50)[::-1]]))
    plt.ylabel("Two way time (s)")
    plt.ylim(400, 0)
    if i == 0:
        plt.title("Network")

    # gather (Ground truth)
    ax = f.add_axes(
        [(6 * (i % 2) + 3.7) / ncol, (nrow - 1 - i // 2) / nrow, 0.9 / ncol, 1 / nrow])  # xmin, ymin, dx, and dy

    plt.imshow((t_gain * labels_array)[ib, 0, :, :].T,
               aspect=0.2, cmap="gray_r", vmin=-VLIM, vmax=VLIM)
    plt.xticks([])
    plt.yticks([])
    plt.ylim(400, 0)
    if i == 0:
        plt.title("Ground\ntruth")

    # gather (diff)
    ax = f.add_axes(
        [(6 * (i % 2) + 4.7) / ncol, (nrow - 1 - i // 2) / nrow, 0.9 / ncol, 1 / nrow])  # xmin, ymin, dx, and dy

    plt.imshow((t_gain * (labels_array - outputs_array))[ib, 0, :, :].T,
               aspect=0.2, cmap="gray_r", vmin=-VLIM, vmax=VLIM)
    plt.xticks([])
    plt.yticks([])
    plt.ylim(400, 0)
    if i == 0:
        plt.title("Difference")

    if i == 0:
        ax = f.add_axes([11.7 / ncol, 3.1 / nrow, 0.15 / ncol, 0.8 / nrow])
        cb = f.colorbar(im0, cax=ax)
        cb.ax.set_ylabel('Velocity ($\mathrm{ms}^{-1}$)')

        ax = f.add_axes([11.7 / ncol, 2.1 / nrow, 0.15 / ncol, 0.8 / nrow])
        cb = f.colorbar(im1, cax=ax, aspect=0.01)
        cb.ax.set_ylabel('Pressure (arb)')

plt.savefig("../report_plots/marmousi_test_12.pdf", bbox_inches='tight', pad_inches=0.01, dpi=300)
plt.show()
