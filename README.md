# Stochastic_PredNet

Data and code repository for the paper 

"Coupling a neural network with a spatial dowscaling procedure to improve probabilistic nowcast for urban rain radars"

submitted for pubblication to MDPI journal : "Remote Sensing"

After installing torch and pysteps library with pip:

pip install torch

pip install pysteps

the notebook: 

Stochastic_PredNet.ipynb

which implements the nowcast methodology described in the paper can be run.
An example of the input and output files obained wiuth the notebook is shown below.


The map of Japan in which is highlighted the area that represents the domain covered by the weather radar whose data are used in this study

![JapanRadarDomain](https://user-images.githubusercontent.com/32863682/182106966-8595738a-b0a7-420e-b9e0-885da39a4149.jpg)

An example of data input: one hour of observations
![obs_ass](https://user-images.githubusercontent.com/32863682/182146554-100d361d-9001-456b-8a4b-35a69f0fd7d8.png)
The same data "assimilated" in the PredNet
![NN_ass](https://user-images.githubusercontent.com/32863682/182146795-dcc32b60-5e7e-4bfc-a838-c31a9a047920.png)
Two corresponding perturbed scenarios
![obs_m0](https://user-images.githubusercontent.com/32863682/182146974-3e53725e-cc26-43f2-9197-1b7196519140.png)
![obs_m1](https://user-images.githubusercontent.com/32863682/182146990-b7771d40-9d5d-4094-bd6b-8eaad74e4a8d.png)
The following one hour of observations
![obs_nwc](https://user-images.githubusercontent.com/32863682/182147209-5460433c-87f2-45d5-bbd8-931262eb0a17.png)
Deterministic PredNet nowcast 
![NN_nwc](https://user-images.githubusercontent.com/32863682/182147390-507c07d6-16ba-4e33-b3ad-504e0f0e97c7.png)
and two ensemble members stochastic PredNet nowcast
![NNm0](https://user-images.githubusercontent.com/32863682/182147510-cac14594-8b8e-41ac-b463-47172bb7337a.png)
![NNm1](https://user-images.githubusercontent.com/32863682/182147522-57ed7840-fdaa-4c4a-9d37-23a719d963d9.png)
