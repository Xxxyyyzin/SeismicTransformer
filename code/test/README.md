This part we introduce how to use the test data to test!

-fault_data: this dataset is similar to the training dataset.

<font size=10>Notes：</font>
<br>
**1. Since the trained neural network is too large (about 435.1MB), there is no way to upload it. 
If you need relevant training network information, you can contact us (yanzinxyz@gmail.com)\~**
<br>
**2. The obtained training network information is placed in the following folder:
`fault_data/server/models/Adam/fault_SeismicTrans_Adam_l1_mean_loss_gain`**
<br><br>
`ST_analysis_single_velocity.py`:use this file to obtain the test result!
<br><br>
If you want to utilize the different fault velocity or source to gain the test result, you can adjust the following code:
<br>
`ib=0`<br>
`isource=0`<br>
`plot_result_SeisTrans(inputs_array, outputs_array100thous, labels_array, title="SeismicTrans\n100000th",sample_batch=None, ib=0, isource=0,aspect=0.2`
<br>Among them,<br>
-   `ib` indicates the number of velocity models, and the value of `ib` is from 0 to 9.
-   `isource` indicates the number of sources, and the value of `isource` is from 0 to 2.
<br>
- marmousi_data: similar to the `fault_data` but is the sliced Marmousi velocity model.
