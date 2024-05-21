"""
This script is: Drawing a single velocity model and its prediction under the AutoEncoder model
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from analysis_Conv import load_model_Conv, load_testdataset_Conv, plot_result_Conv
from criterion_single_function import EvaluationValueALL_single

# Load model and dataset
model, c_dict = load_model_Conv("fault_AE_r_Adam_l1_mean_loss_gain",rootdir="server/",optimizer="Adam",nth_model=4, verbose=False)
d = load_testdataset_Conv("fault_2ms_r_generalisation.bin",rootdir="../generate_data_main/data/",  N_EXAMPLES=30, c_dict=c_dict, verbose=False)
#for k in c_dict: print("%s: %s"%(k, c_dict[k]))

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
with torch.no_grad():  # faster inference without tracking
    model.eval()
    outputs = model(*inputs)

    inputs_array = inputs[
        0].detach().cpu().numpy().copy()  # detach returns a new tensor, detached from the current graph
    outputs_array = outputs[0].detach().cpu().numpy().copy()
    labels_array = labels[0].detach().cpu().numpy().copy()


ib=0
isource=0
plot_result_Conv(inputs_array, outputs_array, labels_array,title="CONV\n100000th", sample_batch=None, ib=4, isource=0,aspect=0.2, T_GAIN=2.5)
# plot_result_withoutDIFF_Conv(inputs_array, outputs_array, labels_array, title="CONV\n100000th",sample_batch=None, ib=0, isource=0,aspect=0.2, T_GAIN=2.5)   #不含差分图

true_gather = labels_array[ib,isource,:,:]
conv_gather = outputs_array[ib,isource,:,:]
value_conv = EvaluationValueALL_single(true_gather,conv_gather,irange)
print("Conv:")
print(value_conv)
plt.savefig("./con_l1.pdf")
