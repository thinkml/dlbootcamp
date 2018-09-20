from tensorflow.keras.datasets import fashion_mnist

from tensorflow.keras.layers import (Conv2D,
                                     Dropout,
                                     Dense,
                                     Flatten,
                                     MaxPooling2D)

from tensorflow.keras.models import Sequential
from tensorflow import keras

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
print(X_train.shape)

X_train = X_train.reshape(60000, 28, 28, 1)
print(X_train.shape)

num_classes = len(set(y_train))

img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

# convert class vectors to binary class matrices

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
batch_size = 10
epochs = 10

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
