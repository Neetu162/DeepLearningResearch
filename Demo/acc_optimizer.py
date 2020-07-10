import pandas
import matplotlib.pyplot as plt


def main():
    #neurons_in = [10, 20, 30, 40]
    optimizers_in = ['nadam','adam','RMSprop', 'SGD']
   
   
    for optimizerVar in optimizers_in:
       
         optimizer_input = open("log_" + optimizerVar + ".csv", 'r')
         
         optimizer_df = pandas.read_csv(optimizer_input)
         
         optimizer_score = optimizer_df.get("accuracy")
         optimizer_epochs = optimizer_df.get("epoch")
         
         
         plt.figure(1)
         plt.plot(optimizer_epochs, optimizer_score, marker='v',linewidth=1, markersize=13, label=optimizerVar)
     
         #plot formatting
         plt.legend(loc='lower right', fontsize = 10)
         plt.xlabel('Epochs', fontsize = 12)
         plt.ylabel('Accuracy', fontsize = 12)
         plt.tick_params(labelsize=14)
         plt.grid()

    plt.show()




if __name__=="__main__":
    main()
