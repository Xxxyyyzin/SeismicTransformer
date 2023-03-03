"""
This script is :Transformer模型下绘制单个速度模型比较，含差分图
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from analysis_SeisTrans import load_model_SeisTrans, load_testdataset_SeisTrans, plot_result_SeisTrans ,plot_result_withoutDIFF_SeisTrans
from criterion_single_function import EvaluationValueALL_single

# Load model and dataset
model100thous, c_dict100thous = load_model_SeisTrans("marmousi_SeismicTrans_Adam_l1_mean_loss_gain", rootdir="server/",optimizer="Adam",nth_model=0,verbose=False)
d = load_testdataset_SeisTrans("marmousi_2ms_r_generalisation.bin",rootdir="./", N_EXAMPLES=30, c_dict=c_dict100thous, verbose=False)   #N_EXAMPLES 表示数据集下有多少个数据

# Get batches of test data
irange = np.arange(10)# Get batches of test data


d.open_file_reader()
samples = [d[i] for i in irange]
d.close_file_reader()
inputs = [torch.cat([sample["inputs"][i].unsqueeze(0) for sample in samples]) for i in range(len(samples[0]["inputs"]))]
labels = [torch.cat([sample["labels"][i].unsqueeze(0) for sample in samples]) for i in range(len(samples[0]["labels"]))]
for i in inputs: print(i.shape)
for i in labels: print(i.shape)

# Inference
with torch.no_grad():  # faster inference without tracking
    model100thous.eval()
    outputs100thous = model100thous(*inputs)

    inputs_array = inputs[0].detach().cpu().numpy().copy()  # detach returns a new tensor, detached from the current graph
    # outputs_array = outputs[0].detach().cpu().numpy().copy()
    labels_array = labels[0].detach().cpu().numpy().copy()
    outputs_array100thous=outputs100thous.numpy()


ib=0
isource=0
plot_result_SeisTrans(inputs_array, outputs_array100thous, labels_array, title="SeismicTrans\n100000th",sample_batch=None, ib=7, isource=0,aspect=0.2, T_GAIN=2.5)
# plot_result_withoutDIFF_SeisTrans(inputs_array, outputs_array100thous, labels_array, sample_batch=None, ib=0, isource=0,aspect=0.2, T_GAIN=2.5)   #不含差分图
# plt.savefig("../report_plots/trans_l1.pdf")
# plt.show()
true_gather=labels_array[ib,isource,:,:]
trans_gather=outputs_array100thous[ib,isource,:,:]
value_trans = EvaluationValueALL_single(true_gather,trans_gather,irange)
print("Trans:")
print(value_trans)
plt.savefig("./marmousi_Seismictransformer_l1.pdf")