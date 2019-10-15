import numpy as np
import tensorflow as tf

tf.set_random_seed(23456)  # reproducibility

class AdamWParameter(object):
    def __init__(self, nEpochs = 200, Te=20, Tmult=2, LR=0.001, weightDecay=0.025, batchSize=32, nBatches = 0):
        self.nEpochs     = nEpochs                #number of total epoch
        self.Te          = Te                     #Ti: total number of epochs within the i-th run / restart of the algorithm
        self.Tmult       = Tmult                  
        self.LR          = LR                     #learning rate
        self.weightDecay = weightDecay            #learning rate decay rate 
        self.batchSize   = batchSize              #bt: batch size                   
        self.nBatches    = nBatches               #number of total batch
        self.EpochNext   = self.Te + 1            #next restart epoch
        self.T_cur       = 0   
        self.t           = 0 
        self.wd          = self.weightDecayNormalized()
        self.H_cur       = 0.9
    
    
    #yita
    def learningRateCosineSGDR(self, epoch):
        self.T_cur = self.T_cur + 1.0 / (self.Te * self.nBatches)
        if self.T_cur >= self.H_cur:
            self.T_cur = self.H_cur
        if (self.T_cur >= self.H_cur) and (epoch == self.EpochNext):
            self.T_cur = 0
            self.Te = self.Te * self.Tmult
            self.EpochNext = self.EpochNext + self.Te
        
        return 0.5 * (1 + np.cos(np.pi * self.T_cur))

    #wt
    def weightDecayNormalized(self):
        return self.weightDecay / (np.power(self.nBatches * self.Te, 0.5))
    
    
    #update and get paramter every epoch
    def getParameter(self, epoch):
        
        yita = self.learningRateCosineSGDR(epoch)
        lr   = yita * self.LR
        clr  = lr/(1+self.t*0)                               #currentLearningRate
        wdc  = yita * self.wd                                #weightDecayCurrent
        self.t +=1 
        return (
                np.float32(clr),
                np.float32(wdc)
                )
               
               
               
               
               
               
               
               
               
               