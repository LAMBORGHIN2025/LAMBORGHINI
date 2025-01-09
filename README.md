# LAMBORGHINI

This repository includes the experimental code for reproducing the results presented in the paper titled "LAMBORGHINI: Low Latency Mix-Nets with Balancing Load Distribution Achieved Through Cost-Minimized Protocols," currently under submission for the first cycle of CCS 2025.

# Initial Setup and Dependencies
The code is designed to run on any standard laptop or desktop with Ubuntu 18.04 or higher and has been tested with Python 3.8.10. The required dependencies are as follows:
-	 `fpdf==1.7.2 ` 
-	 ` kaleido==0.2.1` 
-	 `matplotlib==3.5.2` 
-	 `numpy==1.21.2` 
-	 ` plotly==5.10.0 ` 
-	 ` pulp==2.7.0 ` 
-	 ` scikit_learn==1.1.1 ` 
-	 ` scikit-learn-extra==0.2.0 ` 
-	 ` scipy==1.8.1 ` 
-	`simpy==4.0.1`  

 	To install the required libraries, use the following command:
 ` pip install -r requirements.txt ` 

# Code Execution
The experiments described in the paper can be reproduced using the  ` Main.py file ` . Specify the appropriate Input argument when running Main.py to execute the desired experiment:
-	Figures 2, 3, and 4 (E1, E2, and E3)
- Input: 1
-	Table 1 (E4)
- Input: 2
-	Table 2 (E5)
- Input: 3

# Additional Notes
1.	After running each experiment, the corresponding figures will be automatically saved in the Figures folder. Tables will be displayed in the command-line window.
2.	The parameters in Main.py are preconfigured to reproduce the results consistent with the main body of the paper. You are welcome to modify these parameters if desired.
3.	Increasing the number of iterations in Main.py improves accuracy and reduces sampling errors for better results. However, it also increases execution time. The default number of iterations is set to 50, while the original results in the paper were obtained with 400 iterations.

# Hardware Requirements
The code has been tested to run on commodity hardware with the following specifications:
-	RAM: 16 GB
-	CPU: 8 cores
-	Disk Storage: 50 GB

# Brief Description of Key Scripts
Below is a brief overview of the key scripts included in this repository:
-	 ` Main.py ` : The primary script for running experiments.
-	 ` LAMBORGHINI.py ` : Contains the main functions necessary for executing the LAMBORGHINI experiments.
-	 ` Main_F.py ` : Includes functions to evaluate the effects of LONA, strategic routing, and LBA.
-	 ` Greedy_LARMix.py ` : Implements load-balancing techniques as described in the LARMix paper.
-	 ` ripe_November_12_2023_cleaned.json ` : The RIPE latency dataset used in the experiments.
-	 ` Fancy_Plot.py ` : Contains functions for plotting the figures presented in the paper.
-	 `Sim.py` or `Sim2.py`: Contains the core simulation logic of LAMBORGHINI.
- `NYM.py` or `XRD.py`: Provides the implementation of mix-net architectures for use within the simulations.
- `Mix_Node_.py`: Defines the mix-node objects, including their behavior and interactions within the simulated environment.
- `Message_.py` or `Packets.py`: Implements the structure and functionality of packet (or message) objects for the simulations.
- `GateWay.py`: Acts as a proxy between clients and the mix-net, facilitating communication in the simulation setup.
- `Timed_Mix.py`: Models a timed mix-net object, simulating timing-based delay in the mix-net.

