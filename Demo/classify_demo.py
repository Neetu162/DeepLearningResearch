import pickle
import tempfile
from tensorflow.keras.models import Sequential, load_model, save_model, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dropout, Input, concatenate
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

def unpack(model, weights):
    restored_model = deserialize(model)
    restored_model.set_weights(weights)
    return restored_model

# Hotfix function
def make_keras_picklable():

    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, weights))

    cls = Model
    cls.__reduce__ = __reduce__
make_keras_picklable()
import numpy as np
import time
import pandas


from functools import partial
from nltk.tokenize.regexp import regexp_tokenize
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix



def get_model(model):
    return model

def main():

    good_path = "/home/osboxes/DeepLearningResearch/Data/badging_med/mal_badging_med.txt"

    mal_path = "/home/osboxes/DeepLearningResearch/Data/badging_med/ben_badging_med.txt"

    tr = .80
    neurons = 20
    batch = 100
    epochs = 10

    perm_inputs, feat_inputs, labels = vectorize(good_path, mal_path)
    print("returned from vectorize method" + str(perm_inputs) + "feat Inputs" + str(feat_inputs) + "labels" + str(labels))
    perm_width = int(len(perm_inputs[0]))
    print("perm_width" + str(perm_width))
    feat_width = int(len(feat_inputs[0]))
    print("feat_width" + str(feat_width))
    cm = np.zeros([2,2], dtype=np.int64)
    model = create_dualInputLarge(input_ratio=.125, neurons=neurons, perm_width=perm_width, \
    feat_width=feat_width)
    plot_model(model, to_file='model.png')
   # model.summary()
    time.sleep(10)

    sss = StratifiedShuffleSplit(n_splits=1, random_state=0, test_size=1-tr)
    i = 0
    print("stratified shuffle split")
    for train_index, test_index in sss.split(perm_inputs, labels):
        perm_train, perm_test = perm_inputs[train_index], perm_inputs[test_index]
        feat_train, feat_test = feat_inputs[train_index], feat_inputs[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]
        print ("perm_width: " + str(perm_width))
        print ("feat_width: " + str(feat_width))
        model = create_dualInputLarge(input_ratio=.125, neurons=neurons, perm_width=perm_width, \
        feat_width=feat_width)

        print('\nsplit %i' %i)
        model.fit([perm_train, feat_train], labels_train, epochs=epochs, batch_size=batch)
        print("model trained")
        labels_pred = model.predict([perm_test, feat_test], batch_size=batch)
        print("prediction made: " +str(labels_pred))
        labels_pred = (labels_pred > 0.5)
        print("labels_pred" +str(labels_pred))
        cm = cm + confusion_matrix(labels_test, labels_pred)
        i += 1

    acc = calc_accuracy(cm)
    print ('average accuracy was: ' + str(acc))
    
    precision = calc_precision(cm)
    print('Average precision was: ' + str(precision))
    
    recall = cal_recall(cm)
    print('Average recall value is: ' + str(recall))
    
    
    scoring = ['precision', 'accuracy', 'recall', 'f1']
    perm_inputs_1 = perm_inputs
    print("creating the loaded model")
    loaded_model = KerasClassifier(build_fn=get_model(model))
    print("calling the cross_validate method")
    
    cv_result = cross_validate(loaded_model, perm_inputs_1, labels, cv=5, return_train_score=True, n_jobs=1)
    df = pandas.DataFrame(cv_result)
    
    path1 = '/home/osboxes/DeepLearningResearch/Demo/test' + '.csv'
    file1 = open(path1, "a+")
    df.to_csv(file1, index=True)
    file1.close()
    
    return

def calc_accuracy(cm):
    TP = float(cm[1][1])
    TN = float(cm[0][0])
    n_samples = cm.sum()
    return (TP+TN)/n_samples

def calc_precision(cm):
    TP = float(cm[1][1])
    FP = float(cm[1][0])
    return TP/(TP+FP)

def cal_recall(cm):
    TP = float(cm[1][1])
    FN = float(cm[0][1])
    return TP/(TP + FN)

def vectorize(good_path, mal_path):

    with open(good_path) as f:
        ben_samples = f.readlines()
    with open(mal_path) as f:
        mal_samples = f.readlines()

    samples = ben_samples + mal_samples

    labels = np.array([])
    for x in ben_samples:
        labels = np.append(labels, 0)
    print("benign sample labels" + str(labels))
    for x in mal_samples:
        labels = np.append(labels, 1)
    print("benign + malware sample labels" + str(labels))
    

    perm_pattern = "(?:\w|\.)+(?:permission).(?:\w|\.)+"
    feat_pattern = "(?:\w|\.)+(?:hardware).(?:\w|\.)+"

    perm_vect = CountVectorizer(analyzer=partial(regexp_tokenize, pattern=perm_pattern))
    feat_vect = CountVectorizer(analyzer=partial(regexp_tokenize, pattern=feat_pattern))

    perm_inputs_sparse = perm_vect.fit_transform(samples)
    perm_inputs_dense = perm_inputs_sparse.todense()
    perm_inputs = np.array(perm_inputs_dense)
    print("got perm_inputs")

    feat_inputs_sparse = feat_vect.fit_transform(samples)
    feat_inputs_dense = feat_inputs_sparse.todense()
    feat_inputs = np.array(feat_inputs_dense)
    print("got feat_inputs")

    return perm_inputs, feat_inputs, labels

def create_dualInputLarge(input_ratio, feat_width, perm_width, neurons=32, dropout_rate=0.3):
    '''this model performs additional analysis with layers after concatenation'''
    perm_width=int(perm_width)
    perm_input = Input(shape=(perm_width,), name='permissions_input')
    print("perm input after Input function: "+ str(perm_input))
    
    x = Dense(neurons, activation='relu')(perm_input)
    print("X:" + str(x))
    x = Dropout(dropout_rate)(x)
    print("X:" + str(x))
    x = Dense(neurons, activation='relu')(x)
    print("X:" + str(x))
    feat_input = Input(shape=(feat_width,), name='features_input')
    y = Dense(int(neurons*input_ratio), activation='relu')(feat_input)
    print("y:" + str(y))
    x = concatenate([x, y])
    x = Dense(int((neurons+(neurons*input_ratio))/2), activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(int((neurons+(neurons*input_ratio))/2), activation='relu')(x)
    output = Dense(1, activation='sigmoid', name="output")(x)
    model = Model(inputs=[perm_input, feat_input], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    main()
