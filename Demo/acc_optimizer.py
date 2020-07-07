import pandas
import matplotlib.pyplot as plt


def main():
    #neurons_in = [10, 20, 30, 40]
    optimizers_in = ['nadam','adam','RMSprop', 'SGD']
   
   
    for optimizerVar in optimizers_in:
        #for neuronVar in neurons_in:
        
         optimizer_input = open("log_" + optimizerVar + ".csv", 'r')
         #adam_in = open("log_" + optimizerVar +"_" + str(neuronVar) + ".csv", 'r')
         #RMSprop_in = open("log_" + optimizerVar +"_" + str(neuronVar) + ".csv", 'r')
         #SGD_in = open("log_" + optimizerVar +"_" + str(neuronVar) + ".csv", 'r')
     
         optimizer_df = pandas.read_csv(optimizer_input)
         #adam_df = pandas.read_csv(adam_in)
         #RMS_df = pandas.read_csv(RMSprop_in)
         #SGD_df = pandas.read_csv(SGD_in)
     
         #nadam x and y
         #nadam_neurons = nadam_df.get("param_neurons")
         optimizer_score = optimizer_df.get("accuracy")
         optimizer_epochs = optimizer_df.get("epoch")
         
         #adam x and y
         #adam_neurons = adam_df.get("param_neurons")
         #adam_score = adam_df.get("accuracy")
     
         #RMS x and y
         #RMS_neurons = RMS_df.get("param_neurons")
         #RMS_score = RMS_df.get("accuracy")
     
         #SGD x and y
         #SGD_neurons = SGD_df.get("param_neurons")
         #SGD_score = SGD_df.get("accuracy")
         
         
         #avg_accuracy = (nadam_score + adam_score + RMS_score + SGD_score)/4
     
         plt.figure(1)
         #print("neuronVar" + str(neuronVar))
         #print("average accuracy" + str(avg_accuracy))
         #create line for each optimizer
        # plt.plot(str(neuronVar), adadelta_score, marker='*', linewidth=3, markersize=13,label='adadelta')
        # plt.plot(str(neuronVar), adamax_score ,marker='o', linewidth=3, markersize=13,label='adamax')
         plt.plot(optimizer_epochs, optimizer_score, marker='v',linewidth=1, markersize=13, label=optimizerVar)
         #plt.plot(str(neuronVar), nadam_score, marker='^', linewidth=3, markersize=13,label='nadam')
         #plt.plot(str(neuronVar), RMS_score, marker='s',linewidth=3, markersize=13, label='RMSprop')
         #plt.plot(str(neuronVar), SGD_score, marker='p',linewidth=3, markersize=13, label='SGD')
     
         #plot formatting
         plt.legend(loc='lower right', fontsize = 14)
         #plt.ylim([.92, .95])
         #plt.title('One Layer Model - Optimizer and Neurons', fontsize=28)
         plt.xlabel('Epochs', fontsize = 20)
         plt.ylabel('Accuracy', fontsize = 20)
         plt.tick_params(labelsize=20)
         plt.grid()
         #plt.legend([adadelta, adamax, adam, RMSprop, SGD]) #['adadelta', 'adamax', 'adam','nadam', 'RMSprop,', 'SGD'])


    plt.show()




if __name__=="__main__":
    main()
