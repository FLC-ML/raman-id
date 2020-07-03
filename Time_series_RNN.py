# Time Series Classification using LSTM/RNN
# Tommy Swimmer, Undergraduate Engineering Student
# Fort Lewis College, 07/01/20

# Importing libraries
import tensorflow as tf
from tensorflow.keras.layers import Input, SimpleRNN, LSTM, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

from sklearn.metrics import confusion_matrix
import itertools

import numpy as np
import matplotlib.pyplot as plt # Use pip install to get this.
from matplotlib import cm
import pandas as pd

### Load the data:
# Training data
X_train_loc = './data/X_reference.npy'
y_train_loc = './data/y_reference.npy'
X_train = np.load(X_train_loc)
y_train = np.load(y_train_loc)
# Testing data
X_test_loc = './data/X_test.npy'
y_test_loc = './data/y_test.npy'
X_test = np.load(X_test_loc)
y_test = np.load(y_test_loc)
# Print shape of data loaded.
print("training set shape:", X_train.shape, y_train.shape)
print("test set shape:", X_test.shape, y_test.shape)
# The data is only 2D
X_train = np.expand_dims(X_train, -1) # adds a superficial dimension to convolve
X_test = np.expand_dims(X_test, -1)
print("\nnew x shape:", X_train.shape)

### Build the Model
i = Input(shape=X_train[0].shape)
x = LSTM(128)(i)
x = Dense(30, activation='softmax')(x)

model = Model(i, x)

### Compile and train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
r = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

### Plot loss per iteration
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

### Plot accuracy per iteration
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()

# Plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

p_test = model.predict(X_test).argmax(axis=1)
cm = confusion_matrix(y_test, p_test)
plot_confusion_matrix(cm, list(range(30)))