The Hybrid model code for PDEs with spatiotemporal parameters
The architecture of the hybrid model is attached as follows:
Run Hybrid_Model.py file
Requires Tensorflow, Pandas, Matplotlib
Outputs: The outputs are videos of the spatiotemporal evolution of parameters and output,
Outputs: Plots for temporal evolutions of temporal variables, .csv file names K_lambda.csv output to get the values of temporally varying and spatially lumped parameters

The structure of the hybrid model is shown below:

![Hybrid neural network model](https://github.com/silabrata/Hybrid-Model/assets/154561737/4c7e0717-074f-4dd0-b1fc-2ec211b54272)

The architecture for the hybrid model for the reaction-diffusion system with the two branches is attached as follows:

![Network_Design_1](https://github.com/silabrata/Hybrid-Model/assets/65000218/dc55a44e-7fd1-4bcc-b838-399187f27aa0)

Initial weights for model training training are taken from Branch_1.h5 and Branch_2.h5.
