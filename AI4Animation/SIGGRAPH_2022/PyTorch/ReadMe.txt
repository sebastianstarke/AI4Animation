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
Each data sample is of shape dimensionality J*T*3, where J are number of
joints (i.e. 24) , T is the time window frames (i.e. 121), and 3 are the 
XYZ velocity values transformed into local root space of the character,
which is calculated as ((V_i in R_i) - (V_(i-1) in R_(i-1))) / dt,
where V is the velocity and R is the root transformation at frame i.

Data Sample: J_1 T_-60 X, ..., J_1 T_+60 X, J_1 T_-60 Y, ..., J_1 T_+60 Y ..., J_N T_-60 Z, ..., J_N T_+60 Z