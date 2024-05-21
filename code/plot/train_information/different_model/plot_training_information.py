# -*- coding: utf-8 -*-
"""
@Time ： 2021/9/15 12:06
@Auth ： Xxxyyzin
@File ：main_SeismicTrans.py
@IDE ：PyCharm
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fcn_values = pd.read_table("./Conv.txt")
seismictrans_values = pd.read_table("./SeismicTrans.txt")
dcn_values = pd.read_table("./DCN.txt")
resnet_values = pd.read_table("./ResNet.txt")
restrans_values = pd.read_table("./ResTransformer.txt")

fcn_training = pd.read_table("./training_AE.txt")
seismictrans_training = pd.read_table("./training_SeismicTransformer.txt")
dcn_training = pd.read_table("./training_DCN.txt")
resnet_training = pd.read_table("./training_ResNet.txt")
restrans_training = pd.read_table("./training_ResTransformer.txt")

def training_loss_10000(training_loss):
    training_loss10000=[]
    for i in range(len(training_loss)):
        if i==0:
            value=training_loss.iloc[0]
            training_loss10000.append(value)
        else:
            if (i+1)%10000==0:
                value=training_loss.iloc[i]
                training_loss10000.append(value)
    return training_loss10000

loss_fcn = training_loss_10000(fcn_training)
loss_seismictrans = training_loss_10000(seismictrans_training)
loss_dcn = training_loss_10000(dcn_training)
loss_resnet = training_loss_10000(resnet_training)
loss_restrans = training_loss_10000(restrans_training)


# fcn
fcn_r1 = fcn_values['ratio-l1']
fcn_rendless = fcn_values['ratio-lendless']
fcn_timecost = fcn_values['timecost']

# seismictrans
x = seismictrans_values["nth model"]/10000
seismictrans_r1 = seismictrans_values['ratio-l1']
seismictrans_rendless = seismictrans_values['ratio-lendless']
seismictrans_timecost = seismictrans_values['timecost']

# dcn
dcn_r1 = dcn_values['ratio-l1']
dcn_rendless = dcn_values['ratio-lendless']
dcn_timecost = dcn_values['timecost']

# resnet
resnet_r1 = resnet_values['ratio-l1']
resnet_rendless = resnet_values['ratio-lendless']
resnet_timecost = resnet_values['timecost']

# restrans
restrans_r1 =restrans_values['ratio-l1']
restrans_rendless = restrans_values['ratio-lendless']
restrans_timecost = restrans_values['timecost']

plt.figure(figsize=(24,18))
# (a)
plt.subplot(2,2,1)
plt.plot(loss_fcn,color='red',label="FCN")
plt.plot(loss_seismictrans,color='#4169E1',label="SeismicTrans")
plt.plot(loss_dcn,color="green",label="DCN")
plt.plot(loss_resnet,color="orange",label="ResNet")
plt.plot(loss_restrans,color="brown",label="ResTrans")
plt.legend(loc=1,fontsize=15)
plt.xlabel("(a) Iteration/10thousands",fontsize=20)
plt.xlim(-1,101)
plt.ylabel("Loss",fontsize=20)
plt.title("Train Loss",fontsize=25)

# (b)
y = np.full(50,0.39)
plt.subplot(2,2,2)
plt.plot(x,fcn_timecost,'ro-',color='red',alpha=0.8,linewidth=1,label="FCN")
plt.plot(x,seismictrans_timecost,'ro-',color='#4169E1',alpha=0.8,linewidth=1,label="SesimcTrans")
# ax3.plot(x,dcn_timecost,'ro-',color="green",alpha=0.8,linewidth=1,label="DCN")
plt.plot(x,resnet_timecost,'ro-',color="orange",alpha=0.8,linewidth=1,label="ResNet")
plt.plot(x,restrans_timecost,'ro-',color="brown",alpha=0.8,linewidth=1,label="ResTrans")
plt.plot(x,y,'ro-',color="black",alpha=0.8,linewidth=1,label="Finite Difference")
plt.legend(loc="upper right",fontsize=15)
plt.xlabel("(b) Iteration/10thousands",fontsize=20)
plt.ylabel("Time Cost",fontsize=20)
plt.title("Time Cost",fontsize=25)

# (c)
plt.subplot(2,2,3)
plt.plot(x,fcn_r1,'ro-',alpha=0.8,linewidth=1,label="FCN")
plt.plot(x,seismictrans_r1,'ro-',color='#4169E1',alpha=0.8,linewidth=1,label="SesimcTrans")
# ax1.plot(x,dcn_r1,'ro-',color="green",alpha=0.8,linewidth=1,label="DCN")
plt.plot(x,resnet_r1,'ro-',color="orange",alpha=0.8,linewidth=1,label="ResNet")
plt.plot(x,restrans_r1,'ro-',color="brown",alpha=0.8,linewidth=1,label="ResTrans")
plt.legend(loc="upper right",fontsize=15)
plt.xlabel("(c) Iteration/10thousands",fontsize=20)
plt.ylabel("R_1",fontsize=20)
plt.title("R1",fontsize=25)

# (d)
plt.subplot(2,2,4)
plt.plot(x,fcn_rendless,'ro-',color='red',alpha=0.8,linewidth=1,label="FCN")
plt.plot(x,seismictrans_rendless,'ro-',color='#4169E1',alpha=0.8,linewidth=1,label="SesimcTrans")
# ax2.plot(x,dcn_rendless,'ro-',color="green",alpha=0.8,linewidth=1,label="DCN")
plt.plot(x,resnet_rendless,'ro-',color="orange",alpha=0.8,linewidth=1,label="ResNet")
plt.plot(x,restrans_rendless,'ro-',color="brown",alpha=0.8,linewidth=1,label="ResTrans")
plt.legend(loc="upper right",fontsize=15)
plt.xlabel("(d) Iteration/10thousands",fontsize=20)
plt.ylabel("R_endless",fontsize=20)
plt.title("R Endless",fontsize=25)
plt.savefig("output.pdf",dpi=300)
plt.savefig("output.png",dpi=300)
plt.show()
