# SeismicTransformer

An attention-based method for seismic wavefield simulation

`FCN`：the network design of the FCN and the training code of the FCN.

`Transformer`：the network design of the SeismicTransformer and the training code of the SeismicTransformer.

`generate_data`: generate the training and testing datasets. 

Different from the folder `generate_data_main`, this folder generates all the velocity models first, and finally 
generates the corresponding seismic records with finite difference. But there is a problem with this. When the 
generated data set is too large, the computer cannot store it. For example, to generate 100,000 velocity models, 
the computer memory is not enough during the process of generating the corresponding 100,000 earthquake records.
Therefore, `generate_data_main` improves this problem: generate velocity models and seismic records in batches 
and convert them into binary data. For example, to generate 100,000 "velocity-seismic records", set to generate
1000 pairs each time, generate 1000 velocity and seismic records for the first time, convert them into binary data,
then clear 1000 velocity and seismic records, and then generate 1000 velocity records And seismic records are converted 
into binary data, and this binary data is superimposed on the data generated for the first time.

`generate_data_main`: generate the training and testing datasets. 

`plot`: function library for plot the relevant figures.


`shared_modules`:Tool function library
