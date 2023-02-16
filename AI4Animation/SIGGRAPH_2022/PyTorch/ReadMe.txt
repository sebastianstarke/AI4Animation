============================================================
Supplementary Code for DeepPhase SIGGRAPH 2022 Submission
============================================================

This code contains the implementation of the Periodic Autoencoder (PAE) model
to extract phase manifolds from full-body motion capture data, from which we
train our neural motion controllers and/or use them as features for motion matching.

The 'Dataset' folder contains a small subset of human dancing movements, and
the 'PAE' folder contains the network file. You can simply run the model after
installing the requirements via the command below (Anaconda is recommended):

python Network.py

Requirement Instructions:
conda create -n DeepPhaseSubmission
conda activate DeepPhaseSubmission
conda install python=3.6 numpy scipy matplotlib
conda install -c anaconda scikit-learn
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

The input/output data provided to the model is a matrix where each row
is one data sample containing the joint velocity features at each frame.
Each data sample is of shape dimensionality k*d, where k are the number
of joints (i.e. 26) and d are the 3 XYZ velocity values transformed into 
the local root space of the character, which is calculated as:

V_i = ((P_i in R_i) - (P_(i-1) in R_(i-1))) / dt

where P is the position and R is the root transformation at frame i 
and i-1 respectively. This creates a training data file with shape 
N x k*d where N are the number of frames (rows) and k*d columns.

Data Sample Format: J1X, J1Y, J1Z, ..., JkX, JkY, JkZ

For more information, open the Unity project, open the "AssetPipeline"
editor window and drag&drop the "DeepPhasePipeline" asset into it. Then
pressing the "Process" button will export the training features for the
motion capture scene that is currently open, which means you will need
to have a motion editor with loaded motion assets in your scene. Training
the network will then automatically generate the phase features in form of
P/F/A/B variables which can be imported into Unity and automatically added
to the motion asset files using the DeepPhaseImporter editor window.

Visual step by step tutorial for learning the quadruped motion controller in the demo can be found here: https://www.youtube.com/watch?v=3ASGrxNDd0k
