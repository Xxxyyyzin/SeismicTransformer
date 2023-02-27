"""
This script is :计算多批数据(conv 与trans 模型的对比计算值)，不同训练模型、不同震源、不同batch的评价值函数
并保存
"""
import time
import numpy as np
import torch
from analysis_SeisTrans import load_model_SeisTrans, load_testdataset_SeisTrans
from analysis_Conv import load_model_Conv, load_testdataset_Conv
import pandas as pd


# (a,b) batch

def calulate_in_different_nthmodel_source_ba(nthmodel,isource,a,b,model_interval):
    """
    nthmodel: 加载训练好的第n个模型进行预测
    isource: 用第isource个震源
    a,b: irange = np.arange(a, b) ,irange 测试数据集
    model_interval: 加载模型的间隔
    """
    # Load model and dataset
    model_trans, c_dict_trans = load_model_SeisTrans("fault_SeismicTrans_Adam_l1_mean_loss_gain(t_gain=2.5,multi-head=3)",
                                                                     rootdir="server/", optimizer="Adam", nth_model=nthmodel,
                                                                     verbose=False)
    # Load model and dataset
    model_conv, c_dict_conv = load_model_Conv("fault_AE_r_Adam_l1_mean_loss_gain(t_gain=2.5)",
                                                              rootdir="server/", optimizer="Adam", nth_model=nthmodel,
                                                              verbose=False)
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
        trans_time0 = time.time()
        model_trans.eval()       # Instantiate model
        outputs_trans = model_trans(*inputs)    # Tansformer output
        trans_time1 = time.time()
        outputs_array_trans = outputs_trans.numpy()     # Data type conversion

        conv_time0 = time.time()
        model_conv.eval()
        outputs_conv = model_conv(*inputs)     # Conv output
        conv_time1 = time.time()
        outputs_array_conv = outputs_conv[0].detach().cpu().numpy().copy()

        labels_array = labels[0].detach().cpu().numpy().copy()

    true_gather = labels_array[:, isource, :, :]        # True gather value of the isource of the (a,b) batch
    predict_gather_trans = outputs_array_trans[:, isource, :, :]     # Transformer prediction value of the isource of the (a,b) batch
    predict_gather_conv = outputs_array_conv[:, isource, :, :]      # Conv prediction value of the isource of the (a,b) batch

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

    trans_l1_mean_of_batch=citerion_l1(true_gather,predict_gather_trans)/len(irange)
    trans_lendless_mean_of_batch=citerion_lendless(true_gather,predict_gather_trans)
    trans_lmajority_mean_of_batch=citerion_majority(true_gather,predict_gather_trans)/len(irange)
    ratio_trans_l1_mean_of_batch=ratio_citerion_l1(true_gather,predict_gather_trans)/len(irange)
    ratio_trans_lendless_mean_of_batch=ratio_citerion_lendless(true_gather,predict_gather_trans)
    ratio_trans_lmajority_mean_of_batch=ratio_citerion_majority(true_gather,predict_gather_trans)/len(irange)

    conv_l1_mean_of_batch=citerion_l1(true_gather,predict_gather_conv)/len(irange)
    conv_lendless_mean_of_batch=citerion_lendless(true_gather,predict_gather_conv)
    conv_lmajority_mean_of_batch=citerion_majority(true_gather,predict_gather_conv)/len(irange)
    ratio_conv_l1_mean_of_batch=ratio_citerion_l1(true_gather,predict_gather_conv)/len(irange)
    ratio_conv_lendless_mean_of_batch=ratio_citerion_lendless(true_gather,predict_gather_conv)
    ratio_conv_lmajority_mean_of_batch=ratio_citerion_majority(true_gather,predict_gather_conv)/len(irange)

    trans_timecost=(trans_time1-trans_time0)/len(irange)
    conv_timecost=(conv_time1-conv_time0)/len(irange)

    # 返回字典格式，直接输出字典数据 避免了疯狂用print
    d1=dict()
    d2=dict()
    d1["nth model"]=(nthmodel+1)*model_interval
    d1["trans-l1"]=trans_l1_mean_of_batch; d1["ratio-trans-l1"]=ratio_trans_l1_mean_of_batch
    d1["trans-lendless"]=trans_lendless_mean_of_batch; d1["ratio-trans-lendless"]=ratio_trans_lendless_mean_of_batch
    d1["trans-majority"]=trans_lmajority_mean_of_batch; d1["ratio-trans-majority"]=ratio_trans_lmajority_mean_of_batch
    d1["trans-timecost"]=trans_timecost

    d2["nth model"] = (nthmodel + 1) * model_interval
    d2["conv-l1"]=conv_l1_mean_of_batch; d2["ratio-conv-l1"]=ratio_conv_l1_mean_of_batch
    d2["conv-lendless"]=conv_lendless_mean_of_batch; d2["ratio-conv-lendless"]=ratio_conv_lendless_mean_of_batch
    d2["conv-majority"]=conv_lmajority_mean_of_batch; d2["ratio-conv-majority"]=ratio_conv_lmajority_mean_of_batch
    d2["conv-timecost"]=conv_timecost

    # print("The %s-th iteration transformer model , %s-th source,( %s batch) : l1 value is %s(ratio is %s) ,lendless value is %s (ratio is %s),majority value is %s (ratio is %s), time cost is %s."
    #       %((nthmodel+1)*model_interval,(isource+1),len(irange),trans_l1_mean_of_batch,ratio_trans_l1_mean_of_batch,trans_lendless_mean_of_batch,ratio_trans_lendless_mean_of_batch,
    #         trans_lmajority_mean_of_batch ,ratio_trans_lmajority_mean_of_batch,trans_timecost))
    #
    # print("The %s-th iteration conv model , %s-th source,( %s batch) : l1 value is %s(ratio is %s) ,lendless value is %s(ratio is %s), majority value is %s (ratio is %s), time cost is %s."
    #       %((nthmodel+1)*model_interval,(isource+1),len(irange),conv_l1_mean_of_batch,ratio_conv_l1_mean_of_batch,conv_lendless_mean_of_batch,ratio_conv_lendless_mean_of_batch,
    #         conv_lmajority_mean_of_batch,ratio_conv_lmajority_mean_of_batch,conv_timecost))

    return [str((nthmodel+1)*model_interval),trans_l1_mean_of_batch,ratio_trans_l1_mean_of_batch,trans_lendless_mean_of_batch,
            ratio_trans_lendless_mean_of_batch,trans_lmajority_mean_of_batch,ratio_trans_lmajority_mean_of_batch,trans_timecost,
            conv_l1_mean_of_batch,ratio_conv_l1_mean_of_batch,conv_lendless_mean_of_batch,ratio_conv_lendless_mean_of_batch,
            conv_lmajority_mean_of_batch,ratio_conv_lmajority_mean_of_batch,conv_timecost], d1, d2

    ##  函数的返回值可以是字典形式
    # d=dict()
    # d["nth model"]=(nthmodel+1)*model_interval
    # d["trans-l1"]=trans_l1_mean_of_batch; d["ratio-trans-l1"]=ratio_trans_l1_mean_of_batch
    # d["trans-lendless"]=trans_lendless_mean_of_batch; d["ratio-trans-lendless"]=ratio_trans_lendless_mean_of_batch
    # d["trans-majority"]=trans_lmajority_mean_of_batch; d["ratio-trans-majority"]=ratio_trans_lmajority_mean_of_batch
    # d["trans-timecost"]=trans_timecost
    #
    # d["conv-l1"]=conv_l1_mean_of_batch; d["ratio-conv-l1"]=ratio_conv_l1_mean_of_batch
    # d["conv-lendless"]=conv_lendless_mean_of_batch; d["ratio-conv-lendless"]=ratio_conv_lendless_mean_of_batch
    # d["conv-majority"]=conv_lmajority_mean_of_batch; d["ratio-conv-majority"]=ratio_conv_lmajority_mean_of_batch
    # d["conv-timecost"]=conv_timecost
    #
    # return d


# 循环求多个训练模型的评价值 输出dataframe
columns = ['nth model', 'trans-l1', 'ratio-trans-l1',"trans-lendless","ratio-trans-lendless","trans-majority","ratio-trans-majority","trans-timecost",
                          "conv-l1","ratio-conv-l1","conv-lendless","ratio-conv-lendless","conv-majority","ratio-conv-majority","conv-timecost"]
df = pd.DataFrame(columns=columns)
model_interval=20000
model_num=50
for i in range(0,model_num):
    i_value=calulate_in_different_nthmodel_source_ba(i, 0, 0, 100, model_interval)
    print(i_value[1])
    print(i_value[2])
    df.loc[i]=i_value[0]
    df.to_csv('EvaluationValue_Temp_conv_trans.txt', sep='\t', index=False, header=True)
print(df)
df.to_csv('EvaluationValue_conv_trans.csv',sep=',',index=False,header=True)
df.to_csv('EvaluationValue_conv_trans.txt',sep='\t',index=False,header=True)





