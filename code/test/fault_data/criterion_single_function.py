"""
This script is :计算单个速度模型下，不同训练模型、不同震源的评价值
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from analysis_SeisTrans import load_model_SeisTrans, load_testdataset_SeisTrans
from analysis_Conv import load_model_Conv, load_testdataset_Conv

# ib=3   # i-th velocity 0 1 2 4 5 6 7 8         (trans(l1)<conv(l1),trans(lendless)<conv(lendless))
# # trans(l1)<conv(l1),trans(lendless)>conv(lendless):3
# # trans(l1)>conv(l1),trans(lendless)<conv(lendless):9
# isource=0
# # Load model and dataset
# model100thous_trans, c_dict100thous_trans = load_model_SeisTrans("fault_SeismicTrans_Adam_l1_mean_loss_gain", rootdir="server/",optimizer="Adam",nth_model=0,verbose=False)
# # Load model and dataset
# model100thous_conv, c_dict100thous_conv = load_model_Conv("fault_AE_r_Adam_l1_mean_loss_gain",rootdir="server/",optimizer="Adam",nth_model=0, verbose=False)
# d = load_testdataset_SeisTrans("fault_2ms_r_generalisation.bin",rootdir="../generate_data_main/data/", N_EXAMPLES=30, c_dict=c_dict100thous_trans, verbose=False)   #N_EXAMPLES 表示数据集下有多少个数据
#
# # Get batches of test data
# irange = np.arange(10)# Get batches of test data
#
#
# d.open_file_reader()
# samples = [d[i] for i in irange]
# d.close_file_reader()
# inputs = [torch.cat([sample["inputs"][i].unsqueeze(0) for sample in samples]) for i in range(len(samples[0]["inputs"]))]
# labels = [torch.cat([sample["labels"][i].unsqueeze(0) for sample in samples]) for i in range(len(samples[0]["labels"]))]
# for i in inputs: print(i.shape)
# for i in labels: print(i.shape)
#
# # Inference
# with torch.no_grad():  # faster inference without acking
#     trans_time0=time.time()
#     model100thous_trans.eval()
#     outputs100thous_trans= model100thous_trans(*inputs)
#     trans_time1=time.time()
#     outputs_array100thous_trans = outputs100thous_trans.numpy()
#
#     conv_time0=time.time()
#     model100thous_conv.eval()
#     outputs100thous_conv= model100thous_conv(*inputs)
#     conv_time1=time.time()
#     outputs_array100thous_conv = outputs100thous_conv[0].detach().cpu().numpy().copy()
#
#
#     inputs_array = inputs[0].detach().cpu().numpy().copy()  # detach returns a new tensor, detached from the current graph
#     labels_array = labels[0].detach().cpu().numpy().copy()
#
#
#
# true_gather= labels_array[ib,isource,:,:]
# predict_gather_trans=outputs_array100thous_trans[ib, isource, :, :]
# predict_gather_conv=outputs_array100thous_conv[ib, isource, :, :]



def citerion_l1(true_gather,predict_gather):
    l1=np.sum(np.abs(true_gather-predict_gather))
    return l1

def citerion_lendless(true_gather,predict_gather):
    lendless=np.max(np.abs(true_gather-predict_gather))
    return lendless

def citerion_majority(true_gather,predict_gather):
    true_gather_majority=true_gather[np.where(np.abs(true_gather)>=np.mean(np.abs(true_gather)))]
    corresponding_predict_majority=predict_gather[np.where(np.abs(true_gather)>=np.mean(np.abs(true_gather)))]
    l_majority=np.sum(np.abs(true_gather_majority-corresponding_predict_majority))
    return l_majority


def ratio_citerion_l1(true_gather,predict_gather):
    l1 = np.sum(np.abs(true_gather - predict_gather))
    true_abs_sum=np.sum(np.abs(true_gather))
    ratio_l1=l1/true_abs_sum
    return ratio_l1

def ratio_citerion_lendless(true_gather,predict_gather):
    lendless = np.max(np.abs(true_gather-predict_gather))
    true_max=np.max(true_gather)
    ratio_lendless=lendless/true_max
    return ratio_lendless

def ratio_citerion_majority(true_gather,predict_gather):
    true_gather_majority=true_gather[np.where(np.abs(true_gather)>=np.mean(np.abs(true_gather)))]
    corresponding_predict_majority=predict_gather[np.where(np.abs(true_gather)>=np.mean(np.abs(true_gather)))]
    l_majority=np.sum(np.abs(true_gather_majority-corresponding_predict_majority))
    ratio_lmajority=l_majority/np.sum(np.abs(true_gather_majority))
    return ratio_lmajority

def EvaluationValueALL_single(true_gather,predict_gather,irange):
    """
    不是计算一个batch的就是计算一个速度图的
    """
    if len(true_gather)==32:
        num_dataset=1
    else:
        num_dataset=len(irange)
    l1=citerion_l1(true_gather,predict_gather)/num_dataset
    ratio_l1=ratio_citerion_l1(true_gather,predict_gather)/num_dataset
    lendless=citerion_lendless(true_gather,predict_gather)
    ratio_lendless=ratio_citerion_lendless(true_gather,predict_gather)
    lmajority=citerion_majority(true_gather,predict_gather)/num_dataset
    ratio_lmajority=ratio_citerion_majority(true_gather,predict_gather)/num_dataset
    d=dict()
    d["l1"]=l1;d["ratio_l1"]=ratio_l1
    d["lendless"]=lendless;d["ratio_lendless"]=ratio_lendless
    d["lmajority"]=lmajority;d["ratio_lmajority"]=ratio_lmajority
    return d

#####  单个速度对比
# print("==================================")
# print("trans:")
# value_trans=EvaluationValueALL_single(true_gather,predict_gather_trans,irange)
# print(value_trans)
# print("SeismicTransformer simulation costs: %s second per velocity!"%((trans_time1-trans_time0)/len(irange)))
# print("==================================")
#
# print("convolution:")
# value_conv=EvaluationValueALL_single(true_gather,predict_gather_conv,irange)
# print(value_conv)
# print("Conv simulation costs: %s second per velocity!"%((conv_time1-conv_time0)/len(irange)))


# ##### plot difference distribution
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy import stats, integrate
#
# # 转化为一维数组便于用seaborn画图
# dif1=np.abs(true_gather-predict_gather_trans).reshape(-1)
# dif2=np.abs(true_gather-predict_gather_conv).reshape(-1)
#
# # 删除满足条件的数据，便于画图美观
# new_dif1 = np.delete(dif1, (dif1>0.01)|(dif1<0) )
# new_dif2 = np.delete(dif2, (dif2>0.01)|(dif2<0) )
# # 与以上删除代码等价
# new_dif1=dif1[~((dif1 < 0) | (dif1>0.01))]
# new_dif2=dif2[~((dif1 < 0) | (dif2>0.01))]

# ##### 频率分布图
# sns.distplot(new_dif1,bins=20,kde=False,hist_kws={"color":"steelblue"},label="Trans")
# sns.distplot(new_dif2,bins=20,kde=False,hist_kws={"color":"purple"},label="Conv")
# plt.title("Trans-Conv")
# plt.legend()
# plt.xlabel("Difference")
# plt.xlim(0)
# plt.ylabel("Count")
# plt.show()

# # #### 绘制核密度图
# plt.figure(figsize=(16, 10), dpi=80)
# sns.kdeplot(new_dif1, shade=True, color="dodgerblue", label="Trans", alpha=.7)
# sns.kdeplot(new_dif2, shade=True, color="orange", label="Conv", alpha=.7)
# # Decoration
# plt.title("Trans-Conv", fontsize=22)
# plt.legend()
# plt.xlabel("Difference")
# plt.xlim(0)
# plt.ylabel("Count")
# plt.show()


# ##### 绘制核密度图
# sns.distplot(new_dif1, norm_hist=True ,hist=False, kde_kws={"color": "red", "linestyle": "-"}, label="Trans")
# sns.distplot(new_dif2, norm_hist=True,hist=False, kde_kws={"color": "blue", "linestyle": "--"},  label="Conv")
# # 添加标题
# plt.title("Trans-Conv")
# # 显示图例
# plt.legend()
# plt.xlabel("Difference")
# # plt.xlim(0,0.012)
# plt.ylabel("Count")
# # 显示图形
# plt.show()
