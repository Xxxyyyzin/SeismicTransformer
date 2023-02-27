import pandas as pd

"""分批次训练的损失拼接代码"""


training_1 = pd.read_table("./training1_SeismicTransformer_l1_mean_loss_gain_SGD.txt",sep="\t",header=None)
testing_1 = pd.read_table("./training1_SeismicTransformer_l1_mean_loss_gain_SGD.txt",sep="\t",header=None)

training_2 = pd.read_table("./training2_SeismicTransformer_l1_mean_loss_gain_SGD.txt",sep="\t",header=None)
testing_2 = pd.read_table("./testing2_SeismicTransformer_l1_mean_loss_gain_SGD.txt",sep="\t",header=None)


training_1_cut = training_1[0:220000]
testing_1_cut = testing_1[0:22]

training_2_cut = training_2[0:280000]
testing_2_cut = testing_2[0:28]

training = pd.concat([training_1_cut,training_2_cut])
testing = pd.concat([testing_1,testing_2])


training.to_csv("training_ST_SGD.txt",sep="\t",index=False,header=False)
testing.to_csv("testing_ST_SGD.txt",sep="\t",index=False,header=False)