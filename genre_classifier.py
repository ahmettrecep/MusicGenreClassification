import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

DATASET_PATH = "C:/Users/casper/Desktop/data.json"

def load_data(dataset_path):
    with open(dataset_path, "r") as fp: # r = read
        data = json.load(fp)
    # convert lists into numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    return X, y

def plot_history(history):
    fig, axs = plt.subplots(2)
    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error subplot
    axs[1].plot(history.history["loss"], label="train loss")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


if __name__ == "__main__":

    # load data
    inputs, targets = load_data(DATASET_PATH)

    # split the data into train test sets
    X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.3)

    # build the network architecture
    model = keras.Sequential([
        # input layer
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),

        # 1st hidden layer
        keras.layers.Dense(1024, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)), # how many neurons we want will start with
        keras.layers.Dropout(0.3),

        # 2nd hidden layer ****
        keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.2),

        # 3nd hidden layer
        keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),  # how many neurons we want will start with
        keras.layers.Dropout(0.2),

        #4rd hidden layer
        keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)), # how many neurons we want will start with
        keras.layers.Dropout(0.2),

        #output layer
        keras.layers.Dense(10, activation="softmax") # 10 farklı veri tipimiz olduğu için(müzik türleri) çıkışta 10 nöron olur.
    ])
    # compile network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001) # Adam, gradident descent türevi
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()
    # train network
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=80,
              batch_size=256)
    # plot accuracy and error over the epochs
    plot_history(history)