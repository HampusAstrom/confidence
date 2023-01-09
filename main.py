import os
# Turn off some memory warning by changing warning level before loading module
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from datetime import datetime

from tensorflow import keras
import tensorboard
tensorboard.__version__

print("TensorFlow version: ", tf.__version__)

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

""" This file is adapted from the baseline method from tensorlow """

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#print("GPU devices Available: ", tf.config.list_physical_devices('GPU'))

def cifar10_main():
  """ Model """
  inputs = keras.Input(shape=(32, 32, 3), name="img")
  x = layers.Conv2D(32, 3, activation="relu")(inputs)
  x = layers.Conv2D(64, 3, activation="relu")(x)
  block_1_output = layers.MaxPooling2D(3)(x)

  x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
  x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
  block_2_output = layers.add([x, block_1_output])

  x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
  x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
  block_3_output = layers.add([x, block_2_output])

  x = layers.Conv2D(64, 3, activation="relu")(block_3_output)
  x = layers.GlobalAveragePooling2D()(x)
  flattened_output = layers.Dense(256, activation="relu")(x)
  x = layers.Dropout(0.5)(flattened_output)
  main_outputs = layers.Dense(10)(x)

  """ Confidence network (grafted on) """
  x = layers.Dense(64)(flattened_output)
  x = layers.concatenate([x, main_outputs])
  confidence_output = layers.Dense(1)(x)

  main_model = keras.Model(inputs, main_outputs, name="toy_resnet")
  confidence_model = keras.Model(inputs, confidence_output, name="toy_resnet_confidence")
  # Vizualize
  main_model.summary()
  keras.utils.plot_model(main_model, "mini_resnet.png", show_shapes=True)

  """ Import data and preprocess """
  (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

  x_train = x_train.astype("float32") / 255.0
  x_test = x_test.astype("float32") / 255.0
  y_train = keras.utils.to_categorical(y_train, 10)
  y_test = keras.utils.to_categorical(y_test, 10)

  """ Compile model with loss and optimizer """

  main_model.compile(
      optimizer=keras.optimizers.RMSprop(1e-3),
      loss=keras.losses.CategoricalCrossentropy(from_logits=True),
      metrics=["acc"],
  )

  """ Tensorboard callbacks """
  # Define the Keras TensorBoard callback.
  logdir_main="logs/fit_main/" + datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback_main = keras.callbacks.TensorBoard(log_dir=logdir_main)
  logdir_confidence="logs/fit_confidence/" + datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback_confidence = keras.callbacks.TensorBoard(log_dir=logdir_confidence)

  """ Fit """
  main_model.fit(x_train,
                 y_train,
                 batch_size=64,
                 epochs=20,
                 validation_split=0.2,
                 callbacks=[tensorboard_callback_main])

  """ CONFIDENCE MODEL FITTING """

  """ Freeze base model """

  main_model.trainable = False
  confidence_model.summary()
  keras.utils.plot_model(confidence_model, "mini_resnet_confidence.png", show_shapes=True)

  """ Compile confidence model with loss and optimizer """

  confidence_model.compile(
      optimizer=keras.optimizers.RMSprop(1e-3),
      loss=keras.losses.MeanAbsoluteError(),
      metrics=["acc"],
  )

  """ Produce dataset for confidence by predicting with main network """
  y_pred_train = main_model.predict(x_train)
  true_ind = np.argmax(y_train,axis=1)
  pred_ind = np.argmax(y_pred_train,axis=1)
  y_conf_train = np.zeros(len(pred_ind))
  y_conf_train[true_ind == pred_ind] = 1

  """ Fit """
  confidence_model.fit(x_train,
                       y_conf_train,
                       batch_size=64,
                       epochs=20,
                       validation_split=0.2,
                       callbacks=[tensorboard_callback_confidence])

  """ Check confidence prediction """

  confidence = confidence_model.predict(x_train[:3])
  print(confidence)
  print(y_conf_train[:3])


def fashion_mnist_main():
  # Define the model.
  sel = 3
  if sel == 1:
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
  elif sel == 2:
    model = keras.Sequential()
    model.add(keras.applications.ResNet50(include_top=True,
                  pooling='avg',
                  weights=None,
                  input_shape=(28, 28)))
  else:
    model = MyModel()

  model.compile(
      optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'])

  (train_images, train_labels), _ = keras.datasets.fashion_mnist.load_data()
  train_images = train_images / 255.0

  # Train the model.
  model.fit(
      train_images,
      train_labels,
      batch_size=64,
      epochs=5,
      callbacks=[tensorboard_callback])

""" Base model """
class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)
  # Define the Keras TensorBoard callback.
  logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

""" Base model """
class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

def main():
  """ Import data """
  mnist = tf.keras.datasets.mnist

  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0

  # Add a channels dimension
  x_train = x_train[..., tf.newaxis].astype("float32")
  x_test = x_test[..., tf.newaxis].astype("float32")

  """ Batch and shuffle """
  train_ds = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(10000).batch(32)

  test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

  # Create an instance of the model
  model = MyModel()

  """ Loss and optimizer """
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  optimizer = tf.keras.optimizers.Adam()


  """ Metrics """
  train_loss = tf.keras.metrics.Mean(name='train_loss')
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

  test_loss = tf.keras.metrics.Mean(name='test_loss')
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

  """ Training baseline funcion """
  @tf.function
  def train_step(images, labels):
    with tf.GradientTape() as tape:
      # training=True is only needed if there are layers with different
      # behavior during training versus inference (e.g. Dropout).
      predictions = model(images, training=True)
      loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


  """ Test baseline funcion """
  @tf.function
  def test_step(images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

  """ Main training loop """
  EPOCHS = 5

  for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
      train_step(images, labels)

    for test_images, test_labels in test_ds:
      test_step(test_images, test_labels)

    print(
      f'Epoch {epoch + 1}, '
      f'Loss: {train_loss.result()}, '
      f'Accuracy: {train_accuracy.result() * 100}, '
      f'Test Loss: {test_loss.result()}, '
      f'Test Accuracy: {test_accuracy.result() * 100}'
    )

if __name__ == "__main__":
  with tf.device('/CPU:0'):
    #main()
    #fashion_mnist_main()
    cifar10_main()
