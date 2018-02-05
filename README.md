# Building Autoencoder in LBANN: A CANDLE Example
CANcer Distributed Learning Enviroment ([CANDLE](http://candle.cels.anl.gov/)) is a collaborative project between US DOE national laboratories and National Cancer Institute (NCI) aims at enabling high-performance deep learning in support of the DOE-NCI Cancer project. As a partner, researchers at [LLNL](www.llnl.gov) are developing open-source HPC deep learning toolkit called [LBANN](https://github.com/LLNL/lbann) to support **CANDLE** and other projects. 

[Autoencoder](https://en.wikipedia.org/wiki/Autoencoder) is one of deep learning techniques being explored in the **CANDLE** team. In this blog, I will explain how to build autoencoder of interest to **CANDLE** project within LBANN framework. Examples in this blog was taken from Tensorflow version of similar deep learning network architecture provided by the **CANDLE** research team.

## Autoencoder in LBANN
A network architecture in LBANN is a collection of layers as a sequential list or graph. To build an autoencoder model in LBANN, the user simply describe how the layers are connected in a [model prototext file](https://github.com/LLNL/lbann/tree/develop/model_zoo/models/autoencoder_candle_pilot1), provide training optimization paratemers in the [optimizer prototext file](https://github.com/LLNL/lbann/tree/develop/model_zoo/optimizers), and input data (and labels in case of classification) in the [data reader prototext file ](https://github.com/LLNL/lbann/tree/develop/model_zoo/data_readers). The prototext files provide the flexibility for users to change a number of network and optimization hyperparameters at run time. For example, an LBANN fully connected (also known as linear or inner product in other deep learning toolkits) layer can be described as shown:
 ```
layer {
    index: 8
    parent: 7
    data_layout: "data_parallel"
    fully_connected {
      num_neurons: 5000
      weight_initialization: "glorot_uniform"
      has_bias: true
    }
  }
  
  ```

Most of the attributes are self descriptive and some of them can be changed "on-the-fly". For instance, the glorot_uniform weight initialization scheme can be replaced with other schemes such as uniform, normal, he_normal, he_uniform, glorot_normal and so on.


## Execute LBANN Autoencoder Example on LC
LBANN has a number of prototext files to support the CANDLE project, one example is provided here. Users can leverage on existing examples as deemed fit. To execute available examples on Livermore Computing (LC) machines:
   1. First install LBANN (detailed instructions available [here](https://github.com/LLNL/lbann.git))
   2. Allocate compute resources using SLURM: `salloc -N1 -t 60`
   3. Run a CANDLE test experiment from the main lbann directory using the following command:
 ```
  srun -n48 build/flash.llnl.gov/model_zoo/lbann \
--model=model_zoo/models/autoencoder_candle_pilot1/model_autoencoder_chem_sigmoid.prototext \
--reader=model_zoo/data_readers/data_reader_candle_pilot1.prototext \
--optimizer=model_zoo/optimizers/opt_adagrad.prototext
```
  First epoch training should produce the following results on Flash:
  ```
 --------------------------------------------------------------------------------
[1] Epoch : stats formated [tr/v/te] iter/epoch = [10311/1146/1273]
            global MB = [ 128/ 128/ 128] global last MB = [  38/  75/ 113]
             local MB = [ 128/ 128/ 128]  local last MB = [  38/  75/ 113]
--------------------------------------------------------------------------------
model 0 average reconstruction cost: 0.017193
Model 0 Epoch time: 2133.91s; Mean minibatch time: 0.190349s; Min: 0.11246s; Max: 0.300455s; Stdev: 0.00308939s
``` 
  LBANN performance will vary on a machine to machine basis. Results will also vary, but should not do so significantly. 

## Running on Non-LC Systems
Launch an MPI job using the proper command for your system (srun, mpirun, mpiexec etc), calling the lbann executable found in lbann/build/$YourBuildSys/model_zoo. This executable requires three command line arguments. These arguments are prototext files specifying the model, optimizer and data reader for the execution. Data directories are hardcoded, make sure you have appropriate permission. Models and other hyperparameters can be adjusted by altering appropriate files. 
```
