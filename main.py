import nltk 
# from nltk.stem.lancaster import LancasterStemmer
# nltk.download('punkt')
import time 
# stemmer = LancasterStemmer()

import numpy 
import tflearn 
import tensorflow 
from tensorflow.python.framework import ops

import random 
import json 
import pickle
from snowballstemmer import TurkishStemmer
turkStem = TurkishStemmer()

# with open('my_intents2.json') as file:
#     data = json.load(file)
#     print("merhabaaaaaaaaa")

with open('intents_.json') as file:
        data = json.load(file)
        print("merhabaaaaaaaaa")

try: 
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except: 
    print('saaaaa')
    words = []
    labels = []
    docs_x = []
    docs_y = [] #2.part

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            
            words.extend(wrds) #iki diziyi birleştirir.
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
            
        if intent["tag"] not in labels:
            labels.append(intent["tag"])     
            
    words = [turkStem.stemWord(w.lower()) for w in words if w != "?"] # tüm pattern'ın içerisindeki kelimeleri parçalayıp attık.
    words = sorted(list(set(words)))
    
    labels = sorted(labels)
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]


# until 2-part
for x, doc in enumerate(docs_x):
    bag = []

    wrds = [turkStem.stemWord(w.lower()) for w in doc]
    
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1
    
    training.append(bag)
    output.append(output_row)
    

training = numpy.array(training)
output = numpy.array(output)

# with open("data.pickle","wb") as f:
#     pickle.dump((words, labels, training, output), f)

# 3-part
ops.reset_default_graph()
# tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)


# model.load("model.tflearn")
#batch_size : girdilerin sayısı
model.fit(training, output, n_epoch=1000, batch_size=90, show_metric=True)
model.save("model.tflearn")
