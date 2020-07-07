import pandas
import matplotlib.pyplot as plt


def main():
    algo_in = ['MultinomialNB', 'BernoulliNB', 'DecisionTree', 'Logistic_Regression']
   
    for algoVar in algo_in:
         algo_input = open(algoVar + "_permissions.csv", 'r')
         
         algo_df = pandas.read_csv(algo_input)
         
         algo_score = algo_df.get("avg_acc")
         algo_ratio = algo_df.get("train_ratio")
         
              
         plt.figure(1)
         plt.plot(algo_ratio, algo_score, marker='v',linewidth=1, markersize=13, label=algoVar)
     
         #plot formatting
         plt.legend(loc='lower right', fontsize = 14)
         plt.xlabel('Train Ratio', fontsize = 20)
         plt.ylabel('Accuracy', fontsize = 20)
         plt.tick_params(labelsize=20)
         plt.grid()
 


    plt.show()




if __name__=="__main__":
    main()
