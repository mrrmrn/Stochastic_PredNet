# Stochastic_PredNet

The map of Japan in which is highlighted the area that represents the domain covered by the weather radar whose data are used in this study

![JapanRadarDomain](https://user-images.githubusercontent.com/32863682/182106966-8595738a-b0a7-420e-b9e0-885da39a4149.jpg)

Repository for codes and a sample of data for the paper:

"Coupling a neural network with a spatial dowscaling procedure to improve probabilistic nowcast for urban rain radars"

submitted for pubblication to MDPI journal : "Remote Sensing".

After installing torch and pysteps libraries, for example with:

pip3 install torch torchvision (for references point to: https://pytorch.org/)

pip install pysteps (for references point to: https://pysteps.github.io/)

the notebook: Stochastic_PredNet.ipynb

which implements the nowcast methodology described in the paper can be run.
An example of the input and output files obained with the notebook is shown below.
To get access to the full data set, please contact the authors of the article sending an email to marino.marrocu@crs4.it

Copyright 2022 by marino.marrocu@crs4.it and luca.massidda@crs4.it

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
  

# An example

Input data: one hour of observations
![obs_ass](https://user-images.githubusercontent.com/32863682/182308793-5b4997dd-caa5-4e4a-b164-4dab7fcdd5a9.png)
The same data "assimilated" in the PredNet
![NN_ass](https://user-images.githubusercontent.com/32863682/182146795-dcc32b60-5e7e-4bfc-a838-c31a9a047920.png)
Two corresponding perturbed scenarios
![obs_m0](https://user-images.githubusercontent.com/32863682/182146974-3e53725e-cc26-43f2-9197-1b7196519140.png)
![obs_m1](https://user-images.githubusercontent.com/32863682/182146990-b7771d40-9d5d-4094-bd6b-8eaad74e4a8d.png)
The following one hour of observations
![obs_nwc](https://user-images.githubusercontent.com/32863682/182147209-5460433c-87f2-45d5-bbd8-931262eb0a17.png)
Deterministic PredNet nowcast 
![NN_nwc](https://user-images.githubusercontent.com/32863682/182147390-507c07d6-16ba-4e33-b3ad-504e0f0e97c7.png)
Two ensemble members nowcast obtained with the stochastic PredNet:
![NNm0](https://user-images.githubusercontent.com/32863682/182147510-cac14594-8b8e-41ac-b463-47172bb7337a.png) 
![NNm1](https://user-images.githubusercontent.com/32863682/182147522-57ed7840-fdaa-4c4a-9d37-23a719d963d9.png)
