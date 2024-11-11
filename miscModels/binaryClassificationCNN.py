from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D, Dropout, GRU, Reshape
from tensorflow.keras.regularizers import l2
# from keras.layers.recurrent import GRU

from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import numpy as np
import matplotlib.pyplot as plt
# USE RNN/lstm
class BinaryClassificationCNN():
    def __init__(self, data):
        self.data = data
        self.inputs, self.targets = self.prepare_data()
        print("inputs",self.inputs.shape)
    def prepare_data(self):
        inputs = np.array(self.data["mfcc"])
        inputs = np.array(inputs.reshape(inputs.shape[0], 94, 13, 1))
        targets = np.array(self.data["Stutter"])
        return inputs, targets
    

    def create_model(self):
        # print("Shape",self.inputs.shape)
        input_shape = (94, 13, 1)
        # X_data = np.array(self.inputs.reshape(self.inputs.shape[0], 3, 44, 1))
        model = Sequential([
            
            Conv2D(32,(8,1), strides=(1,1), input_shape=input_shape, activation='relu'),
            Conv2D(32,(8,1), strides=(1,1), input_shape=input_shape, activation='relu'),
            # Conv2D(32,(8,1), strides=(1,1), input_shape=input_shape, activation='relu'),
            # Conv2D(32,(8,1), strides=(1,1), input_shape=input_shape, activation='relu'),
            # Conv2D(32,(8,1), strides=(1,1), input_shape=input_shape, activation='relu'),
            Reshape((1040,32)),
            GRU(32, return_sequences=True, name='gru1'),
            GRU(32, return_sequences=False, name='gru2'),
            # Flatten(),
            Dropout(0.3),
            Dense(1,activation='sigmoid')

        ])
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer,
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
        return model
    
    def train_model(self, epochs=25, batch_size=64):
    
        inputs_train, inputs_test, targets_train, targets_test = train_test_split(self.inputs, self.targets, test_size=0.2)

        # print(inputs_train.shape, inputs_test.shape, targets_train.shape, targets_test.shape)

        model = self.create_model()
        model.summary()

        hist = model.fit(inputs_train, targets_train, validation_split=0.2, epochs=epochs, batch_size=batch_size)

        # validation_data=(inputs_test,targets_test)
        score = model.evaluate(x=inputs_test,y=targets_test)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        # Predicting
        predicted = model.predict(inputs_test)
        predicted = np.array([1 if x >= 0.5 else 0 for x in predicted])
        actual = np.array(targets_test)
        conf_mat = confusion_matrix(actual, predicted)
        displ = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        displ.plot()

        return hist
    


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


    