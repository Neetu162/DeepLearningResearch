import pandas
import matplotlib.pyplot as plt


def main():
    epoch_acc = open("log_nadam.csv", 'r')
    
    epochAcc_df = pandas.read_csv(epoch_acc)
    

    #nadam x and y
    epochLog = epochAcc_df.get("epoch")
    print("epoch log" + str(epochLog))
    accLog = epochAcc_df.get("accuracy")
    print("accuracy log" + str(accLog))

    
    plt.figure(6)
    
    plt.plot(epochLog, accLog, marker='*', linewidth=1, markersize=13,label='Accuracy VS epoch')
    
    #plot formatting
    plt.legend(loc='lower right', fontsize = 18)
    #plt.ylim([.92, .95])
    #plt.title('One Layer Model - Optimizer and Neurons', fontsize=28)
    plt.xlabel('epoch', fontsize = 20)
    plt.ylabel('Accuracy', fontsize = 20)
    plt.tick_params(labelsize=20)
    plt.grid()
    #plt.legend([adadelta, adamax, adam, RMSprop, SGD]) #['adadelta', 'adamax', 'adam','nadam', 'RMSprop,', 'SGD'])


    plt.show()




if __name__=="__main__":
    main()
