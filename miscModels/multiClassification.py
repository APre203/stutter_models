from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D, Dropout, LSTM, GRU
from tensorflow.keras.regularizers import l2

from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt



class MultiClassification():
    def __init__(self, data):
        self.data = data
        self.mfcc, self.total = self.prepare_data()

    def prepare_data(self):
        mfcc = np.array(self.data["mfcc"])
        total = np.array(self.data["total"])
        return mfcc, total
    
    def create_model(self):
        
        model = Sequential([
        #     LSTM(64, input_shape=(94, 13), return_sequences=True),
            # Flatten(input_shape=(self.mfcc.shape[1],self.mfcc.shape[2])),
            GRU(units=32, dropout=0.2, recurrent_dropout=0.2),
        #     # 1st Hidden Layer
        #     Dense(128, activation="relu", kernel_regularizer=l2(0.001)),
        #     Dropout(0.3),

        #     # 2nd Hidden Layer
        #     Dense(64, activation="relu", kernel_regularizer=l2(0.01)),
        #     Dropout(0.3),
        #     # 3rd Hidden Layer
        #     Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
        #     # Dropout(0.3),
        #     Dense(16, activation="relu", kernel_regularizer=l2(0.001)),

            Dense(6, activation="softmax")

        ])
        # model = Sequential()
        # model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
        # model.add(Dense(6, activation="softmax"))

        # optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer='adam',
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
        return model


    def train_model(self, epochs=100, batch_size=64):
    
        inputs_train, inputs_test, targets_train, targets_test = train_test_split(self.mfcc, self.total, test_size=0.2)

        model = self.create_model()
        # model.summary()

        hist = model.fit(inputs_train, targets_train, validation_data=(inputs_test,targets_test), epochs=epochs, batch_size=batch_size)

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