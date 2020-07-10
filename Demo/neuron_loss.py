import pandas
import matplotlib.pyplot as plt


def main():
    neurons_in = [10, 20, 30, 40]
    #optimizers_in = ['nadam','adam','RMSprop', 'SGD']
   
    #for optimizerVar in optimizers_in:
    for neuronVar in neurons_in:
         neuron_input = open("log_" + str(neuronVar) + ".csv", 'r')
         neuron_df = pandas.read_csv(neuron_input)
         
         neuron_score = neuron_df.get("loss")
         neuron_epochs = neuron_df.get("epoch")
         
                  
     
         plt.figure(4)
         plt.plot(neuron_epochs, neuron_score, marker='v',linewidth=1, markersize=13, label=str(neuronVar) + " neurons")
 
        #plot formatting
         plt.legend(loc='upper right', fontsize = 10)
         plt.xlabel('Epochs', fontsize = 14)
         plt.ylabel('Loss', fontsize = 14)
         plt.tick_params(labelsize=12)
         plt.grid()
         
    plt.show()




if __name__=="__main__":
    main()
