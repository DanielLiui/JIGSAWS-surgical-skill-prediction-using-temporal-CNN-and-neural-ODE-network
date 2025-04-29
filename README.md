# Surgical skill prediction on the JIGSAWS dataset using a temporal CNN and neural ODE
Co-author: Jason Au

The purpose of this study was to explore the ability of neural ordinary differential equations (neural ODEs) to model the changing dynamics of data. A temporal convolution neural network (TCN) with a neural ODE block was implemented to predict the skill level (expert vs non-expert) of surgeons given the kinematics of their tooltip poses using the Da Vinci Robotic system over time. This data comes from the JIGSAWS dataset. The model achieved 100% test accuracy on the knot tying data, 97% on the suturing data, and 80% on the needle-passing data. 

The model is also available for download. 

Dataset:
https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/

<br>

## Installing packages
Can activate program packages by activating virtual environment however that is done in in your IDE. 

If you want to install packages on your local device, run:
```
> pip install -r requirements.txt
```

<br>

## Downloading dataset
1. Download the dataset from https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/ 
2. Make sure the root folder is called 'dataset'. If not, rename it to 'dataset' as the code makes references from this folder.
3. The root folder contains folders called Experimental_setup, Knot_Tying, Needle_Passing, and Suturing that contains data for each suturing task. Within each folder is a readme.txt file that explains what kind of data each folder and file contains. Please read this. This project will just use the kinematic data and meta files to extract skill levels.

<br>

## How to run
There are 2 model training files you can run. You can run main.py, whose main model training function is TCNWithNeuralODE_model(), or you can run the code blocks in main.ipynb in order.  
