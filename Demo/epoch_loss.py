#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 06:25:58 2020

@author: osboxes
"""


import pandas
import matplotlib.pyplot as plt


def main():
    epoch_loss = open("log_nadam.csv", 'r')
    
    epochloss_df = pandas.read_csv(epoch_loss)
    

    #nadam x and y
    epochLog = epochloss_df.get("epoch")
    print("epoch log" + str(epochLog))
    lossLog = epochloss_df.get("loss")
    print("loss log" + str(lossLog))

    
    plt.figure(5)
    
    plt.plot(epochLog, lossLog, marker='*', linewidth=1, markersize=13,label='Epoch VS loss')
      
    #plot formatting
    plt.legend(loc='upper right', fontsize = 18)
    #plt.ylim([.92, .95])
    #plt.title('One Layer Model - Optimizer and Neurons', fontsize=28)
    plt.xlabel('epoch', fontsize = 20)
    plt.ylabel('Loss', fontsize = 20)
    plt.tick_params(labelsize=20)
    plt.grid()
    #plt.legend([adadelta, adamax, adam, RMSprop, SGD]) #['adadelta', 'adamax', 'adam','nadam', 'RMSprop,', 'SGD'])


    plt.show()




if __name__=="__main__":
    main()
