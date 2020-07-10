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
    #plt.legend(loc='lower right', fontsize = 10)
    plt.xlabel('epoch', fontsize = 14)
    plt.ylabel('Accuracy', fontsize = 14)
    plt.tick_params(labelsize=12)
    plt.grid()
    


    plt.show()




if __name__=="__main__":
    main()
