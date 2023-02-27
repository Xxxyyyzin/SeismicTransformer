"""
This script is :一个模型不同的迭代次数下，不同训练模型、不同震源、不同batch的评价值函数
并保存
"""
import time
import numpy as np
import torch
from analysis_SeisTrans import load_model_SeisTrans, load_testdataset_SeisTrans
import pandas as pd


# (a,b) batch
def calulate_in_different_nthmodel_source_ba(nthmodel,isource,a,b,model_interval):
    """
    nthmodel: 加载训练好的第n个模型进行预测
    isource: 用第isource个震源
    a,b: irange = np.arange(a, b) ,irange 测试数据集
    model_interval: 加载模型的间隔
    """
    # Load model
    model, c_dict_trans = load_model_SeisTrans("fault_SeismicTrans_Adam_l1_mean_loss_gain(t=2.5,multi=3,depth=12,lr=0.0001)",
                                                                     rootdir="server/", optimizer="Adam", nth_model=nthmodel,
                                                                     verbose=False)
    # Load  dataset
    d = load_testdataset_SeisTrans("fault_2ms_r_generalisation.bin", rootdir="../generate_data_main/data/",
                                   N_EXAMPLES=300, c_dict=c_dict_trans,
                                   verbose=False)  # N_EXAMPLES 表示数据集下有多少个数据

    # Get batches of test data
    irange = np.arange(a, b)  # Get batches of test data

    d.open_file_reader()
    samples = [d[i] for i in irange]
    d.close_file_reader()
    inputs = [torch.cat([sample["inputs"][i].unsqueeze(0) for sample in samples]) for i in
              range(len(samples[0]["inputs"]))]
    labels = [torch.cat([sample["labels"][i].unsqueeze(0) for sample in samples]) for i in
              range(len(samples[0]["labels"]))]
    for i in inputs: print(i.shape)
    for i in labels: print(i.shape)

    # Inference
    with torch.no_grad():  # faster inference without acking
        time0 = time.time()
        model.eval()       # Instantiate model
        outputs = model(*inputs)    # Tansformer output
        time1 = time.time()
        outputs_array = outputs.numpy()     # Data type conversion


        labels_array = labels[0].detach().cpu().numpy().copy()

    true_gather = labels_array[:, isource, :, :]        # True gather value of the isource of the (a,b) batch
    predict_gather = outputs_array[:, isource, :, :]     # Transformer prediction value of the isource of the (a,b) batch


    def citerion_l1(true_gather, predict_gather):
        l1 = np.sum(np.abs(true_gather - predict_gather))
        return l1

    def citerion_lendless(true_gather, predict_gather):
        lendless = np.max(np.abs(true_gather-predict_gather))
        return lendless

    def citerion_majority(true_gather, predict_gather):
        true_gather_majority = true_gather[np.where(np.abs(true_gather) >= np.mean(np.abs(true_gather)))]
        corresponding_predict_majority = predict_gather[np.where(np.abs(true_gather) >= np.mean(np.abs(true_gather)))]
        l_majority = np.sum(np.abs(true_gather_majority - corresponding_predict_majority))
        return l_majority

    def ratio_citerion_l1(true_gather, predict_gather):
        l1 = np.sum(np.abs(true_gather - predict_gather))
        true_abs_sum = np.sum(np.abs(true_gather))
        ratio_l1 = l1 / true_abs_sum
        return ratio_l1

    def ratio_citerion_lendless(true_gather, predict_gather):
        lendless = np.max(np.abs(true_gather-predict_gather))
        true_max = np.max(true_gather)
        ratio_lendless = lendless / true_max
        return ratio_lendless

    def ratio_citerion_majority(true_gather, predict_gather):
        true_gather_majority = true_gather[np.where(np.abs(true_gather) >= np.mean(np.abs(true_gather)))]
        corresponding_predict_majority = predict_gather[np.where(np.abs(true_gather) >= np.mean(np.abs(true_gather)))]
        l_majority = np.sum(np.abs(true_gather_majority - corresponding_predict_majority))
        ratio_lmajority = l_majority / np.sum(np.abs(true_gather_majority))
        return ratio_lmajority

    l1_mean_of_batch=citerion_l1(true_gather,predict_gather)/len(irange)
    lendless_mean_of_batch=citerion_lendless(true_gather,predict_gather)
    lmajority_mean_of_batch=citerion_majority(true_gather,predict_gather)/len(irange)
    ratio_l1_mean_of_batch=ratio_citerion_l1(true_gather,predict_gather)/len(irange)
    ratio_lendless_mean_of_batch=ratio_citerion_lendless(true_gather,predict_gather)
    ratio_lmajority_mean_of_batch=ratio_citerion_majority(true_gather,predict_gather)/len(irange)


    timecost=(time1-time0)/len(irange)


    # 返回字典格式，直接输出字典数据 避免了疯狂用print
    d1=dict()
    d1["nth model"]=(nthmodel+1)*model_interval
    d1["trans-l1"]=l1_mean_of_batch; d1["ratio-trans-l1"]=ratio_l1_mean_of_batch
    d1["trans-lendless"]=lendless_mean_of_batch; d1["ratio-trans-lendless"]=ratio_lendless_mean_of_batch
    d1["trans-majority"]=lmajority_mean_of_batch; d1["ratio-trans-majority"]=ratio_lmajority_mean_of_batch
    d1["trans-timecost"]=timecost


    return [str((nthmodel+1)*model_interval),l1_mean_of_batch,ratio_l1_mean_of_batch,lendless_mean_of_batch,
            ratio_lendless_mean_of_batch,lmajority_mean_of_batch,ratio_lmajority_mean_of_batch,timecost], d1


# 循环求多个训练模型的评价值 输出dataframe
columns = ['nth model', 'l1', 'ratio-l1',"lendless","ratio-lendless","majority","ratio-majority","timecost"]
df = pd.DataFrame(columns=columns)
model_interval=20000
model_num=50
for i in range(0,model_num):
    i_value=calulate_in_different_nthmodel_source_ba(i, 0, 0, 100, model_interval)
    print(i_value[1])
    df.loc[i]=i_value[0]
    df.to_csv('EvaluationValue_single_Temp(l1,t2.5,m3,d12,lr=0.0001).txt', sep='\t', index=False, header=True)
print(df)
df.to_csv('EvaluationValue_single(l1,t2.5,m3,d12,lr=0.0001).csv',sep=',',index=False,header=True)
df.to_csv('EvaluationValue_single(l1,t2.5,m3,d12,lr=0.0001).txt',sep='\t',index=False,header=True)