from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, auc, roc_curve
import tensorflow.keras as keras

from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

import soundfile as sf
import os
from preprocess_data import mfcc_converter
# USE RNN/lstm
class WordRepModel():
    def __init__(self, data, compare):
        self.data = data
        self.compare = compare
        self.inputs, self.targets = self.prepare_data()
    
    def prepare_data(self):
        inputs = np.array(self.data["mfcc"])
        #inputs = np.array(inputs.reshape(inputs.shape[0], inputs.shape[1], inputs.shape[2], 1))
        targets = np.array(self.data[self.compare])
        return inputs, targets
    

    def create_model(self):
        
        model = Sequential([
        Flatten(input_shape=(self.inputs.shape[1],self.inputs.shape[2])),

        # 1st Hidden Layer
        Dense(512, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.3),

        # 2nd Hidden Layer
        Dense(256, activation="relu", kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        # 3rd Hidden Layer
        Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
        # Dropout(0.3),
        Dense(32, activation="relu", kernel_regularizer=l2(0.001)),

        Dense(2, activation="softmax")

        ])
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
        return model
    
    def train_model(self, epochs=100, batch_size=64):
    
        inputs_train, inputs_test, targets_train, targets_test = train_test_split(self.inputs, self.targets, test_size=0.2)
        #print(inputs_train.shape, inputs_test.shape, targets_train.shape, targets_test.shape)
        model = self.create_model()
        model.summary()
        print(inputs_train.shape)
        print(targets_train.shape)
        print(inputs_test.shape)
        print(targets_test.shape)
        hist = model.fit(inputs_train, targets_train, validation_data=(inputs_test,targets_test), epochs=100, batch_size=64)
        return
        
        score = model.evaluate(x=inputs_test,y=targets_test)
        print('Test loss:', score[0])
        print('Test AUC:', score[1])

        # Predicting
        predicted = model.predict(inputs_test)
        predicted = np.array([1 if x >= 0.5 else 0 for x in predicted])
        actual = np.array(targets_test)

        conf_mat = confusion_matrix(actual, predicted)
        displ = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        displ.plot()
        # auc_result = auc(actual, predicted)
        # print("AUC", auc_result)

        # self.plot_auc(actual, predicted)

        return hist, actual, predicted, model
    
    def plot_hist(self, hist):
        fig, axs = plt.subplots(2)
        # accuracy subplot
        axs[0].plot(hist.history["accuracy"], label="train accuracy")
        axs[0].plot(hist.history["val_accuracy"], label="test accuracy")
        axs[0].set_ylabel("Accuracy")
        axs[0].legend(loc="lower right")
        axs[0].set_title("Accuracy eval")


        # error subplot
        axs[1].plot(hist.history["loss"], label="train loss")
        axs[1].plot(hist.history["val_loss"], label="test loss")
        axs[1].set_ylabel("Loss")
        axs[1].set_xlabel("Epoch")
        axs[1].legend(loc="upper right")
        axs[1].set_title("Loss eval")

        plt.show()


    def plot_auc(self, actual, predicted):
        fpr, tpr, threshold = roc_curve(actual, predicted)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr,tpr)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title('ROC Curve')
        plt.show()

        
