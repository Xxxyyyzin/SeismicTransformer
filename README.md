# SeismicTransformer

An attention-based method for seismic wavefield simulation

<font color=red>It is recommended to run the code under Linux</font>：
1. This code is written under the Linux system.
2. The `.f90` file inside is more convenient for terminal compilation under Linux. (There will be some problems with Windows system)

- FCN ：the network design of the FCN and the training code of the FCN.
`FCN_constants.py`: Basic parameters for FCN model training
`FCN_main.py`: Train the main program
`FCN_models.py`: FCN model architecture
`losses.py`: Custom loss function
`datasets.py`, `mysampler.py`, `torch_utils.py`: The dataset is poured into correlation, you don't need to understand but you need to carefully see if you need to change the **path** !


- Transformer：：the network design of the SeismicTransformer and the training code of the SeismicTransformer.

- generate_data：: generate the training and testing datasets. 


Different from the folder `generate_data_main`, this folder generates all the velocity models first, and finally 
generates the corresponding seismic records with finite difference. But there is a problem with this. When the 
generated data set is too large, the computer cannot store it. For example, to generate 100,000 velocity models, 
the computer memory is not enough during the process of generating the corresponding 100,000 earthquake records.
Therefore, `generate_data_main` improves this problem: generate velocity models and seismic records in batches 
and convert them into binary data. For example, to generate 100,000 "velocity-seismic records", set to generate
1000 pairs each time, generate 1000 velocity and seismic records for the first time, convert them into binary data,
then clear 1000 velocity and seismic records, and then generate 1000 velocity records And seismic records are converted 
into binary data, and this binary data is superimposed on the data generated for the first time.

`add_fault.py`: Add faults
`convert_to_flat_binary_autoencoder.py`: Converted to binary data
`generate_1D_traces.py`: Generate a one-dimensional velocity model
`generate_forward_simulations.py`: Finite difference fractional values simulate relevant parameter control
`generate_velocity_models.py`: Generate a tomographic velocity model
`modified_seismic_CPML_2D_pressure_second_order.f90`: Finite difference fraction value simulates the Fortran code

- generate_data_main: generate the training and testing datasets. 

- plot: function library for plot the relevant figures.
The `train_information` folder is a plot of training loss and evaluation parameters.

The `.py` file under the `plot` folder is mainly a plot of the **dataset** and the **predicted seismic record**:

`criterion_*.py`: Calculate evaluation parameters

`ST_*.py`: plotting predicted seismic records under the SeismicTransformer model
  
`autoencoder_plot.py`: plot predicted seismic records under the FCN model, single map
 
`AE_different_iteration.py`: plot the predicted seismic records under the FCN model with different iterations

`dataset_plot.py`: Dataset plotting

- shared_modules:Tool function library