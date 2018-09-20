from tensorflow.keras.datasets import fashion_mnist

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
print(X_train.shape)

X_train = X_train.reshape(60000, 784)


# One-hot encoding!
# y_train_onehot = np_utils.to_categorical(y_train, 10) !ordered


num_classes = len(set(y_train))
y_train = to_categorical(y_train, num_classes)

assert y_train[0].shape[0] == num_classes

# Model
model = Sequential()
model.add(Dense(num_classes, input_dim=784, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Model Summary
model.summary()


# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=10)


# Evaluate
scores = model.evaluate(X_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
