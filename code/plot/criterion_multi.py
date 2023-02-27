"""
This script is :计算多批数据，不同训练模型、不同震源、不同batch的评价值    ！未封装成函数
"""
import time
import numpy as np
import torch
from analysis_SeisTrans import load_model_SeisTrans, load_testdataset_SeisTrans
from analysis_Conv import load_model_Conv, load_testdataset_Conv

ib=0
isource=0
# Load model and dataset
model100thous_trans, c_dict100thous_trans = load_model_SeisTrans("fault_SeismicTrans_Adam_l1_mean_loss_gain", rootdir="server/",optimizer="Adam",nth_model=0,verbose=False)
# Load model and dataset
model100thous_conv, c_dict100thous_conv = load_model_Conv("fault_AE_r_Adam_l1_mean_loss_gain",rootdir="server/",optimizer="Adam",nth_model=0, verbose=False)
d = load_testdataset_SeisTrans("fault_2ms_r_generalisation.bin",rootdir="../generate_data_main/data/", N_EXAMPLES=300, c_dict=c_dict100thous_trans, verbose=False)   #N_EXAMPLES 表示数据集下有多少个数据

# Get batches of test data
irange = np.arange(100)# Get batches of test data


d.open_file_reader()
samples = [d[i] for i in irange]
d.close_file_reader()
inputs = [torch.cat([sample["inputs"][i].unsqueeze(0) for sample in samples]) for i in range(len(samples[0]["inputs"]))]
labels = [torch.cat([sample["labels"][i].unsqueeze(0) for sample in samples]) for i in range(len(samples[0]["labels"]))]
for i in inputs: print(i.shape)
for i in labels: print(i.shape)

# Inference
with torch.no_grad():  # faster inference without acking
    trans_time0=time.time()
    model100thous_trans.eval()
    outputs100thous_trans= model100thous_trans(*inputs)
    trans_time1=time.time()
    outputs_array100thous_trans = outputs100thous_trans.numpy()

    conv_time0=time.time()
    model100thous_conv.eval()
    outputs100thous_conv= model100thous_conv(*inputs)
    conv_time1=time.time()
    outputs_array100thous_conv = outputs100thous_conv[0].detach().cpu().numpy().copy()


    inputs_array = inputs[0].detach().cpu().numpy().copy()  # detach returns a new tensor, detached from the current graph
    labels_array = labels[0].detach().cpu().numpy().copy()



true_gather= labels_array[:,isource,:,:]
predict_gather_trans=outputs_array100thous_trans[:, isource, :, :]
predict_gather_conv=outputs_array100thous_conv[:,isource, :, :]


def citerion_l1(true_gather,predict_gather):
    l1=np.sum(np.abs(true_gather-predict_gather))
    return l1

def citerion_lendless(true_gather,predict_gather):
    lendless = np.max(np.abs(true_gather-predict_gather))
    return lendless

print("==================================")
print("trans")
l1_trans=citerion_l1(true_gather,predict_gather_trans)
lendless_trans=citerion_lendless(true_gather,predict_gather_trans)
perIrange_l1_trans=l1_trans/len(irange)
perIrange_lendless_trans=lendless_trans/len(irange)
print("The L1 value of SeismicTransformer is(per %s) : %s."%(len(irange),perIrange_l1_trans))
print("The L-endless value of SeismicTransformer is(per %s)  : %s."%(len(irange),perIrange_lendless_trans))
print("SeismicTransformer simulation costs(per %s): %s second!"%(len(irange),trans_time1-trans_time0))
print("==================================")

print("convolution")
l1_conv=citerion_l1(true_gather,predict_gather_conv)
lendless_conv=citerion_lendless(true_gather,predict_gather_conv)
perIrange_l1_conv=l1_conv/len(irange)
perIrange_lendless_conv=lendless_conv/len(irange)
print("The L1 value of Conv is(per %s) : %s."%(len(irange),perIrange_l1_conv))
print("The L-endless value of Conv is(per %s) : %s."%(len(irange),perIrange_lendless_conv))
print("Conv simulation costs(per %s): %s second!"%(len(irange),conv_time1-conv_time0))