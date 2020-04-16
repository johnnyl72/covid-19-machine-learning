import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import tensorflow_docs.plots
import tensorflow_docs.modeling
import tensorflow_docs as tfdocs

print(tf.__version__)

raw_dataset = pd.read_csv("4.14.20_corona_data.csv", sep=",", skipinitialspace=True)
raw_dataset = raw_dataset[["Total Count Confirmed", "Total Count Deaths"]]
dataset = raw_dataset.copy()

dataset.isna().sum()

dataset = dataset.dropna()
print(dataset.head())
print(dataset.tail())

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

sns.pairplot(train_dataset[["Total Count Confirmed", "Total Count Deaths"]], diag_kind="kde")

train_stats = train_dataset.describe()
train_stats.pop("Total Count Confirmed")
train_stats = train_stats.transpose()
train_stats

train_labels = train_dataset.pop('Total Count Confirmed') # Will train this 'y'
test_labels = test_dataset.pop('Total Count Confirmed') # Our test 'y' we will use to compare with

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()
model.summary()

EPOCHS = 2000
history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[tfdocs.modeling.EpochDots()])

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

plotter.plot({'Basic': history}, metric = "mae")
plt.ylim([0, 1500])
plt.ylabel('MAE [Total Count Confirmed]')

plotter.plot({'Basic': history}, metric = "mse")
plt.ylim([0, 2020000])
plt.ylabel('MAE [Total Count Confirmed]')

model = build_model()
# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
early_history = model.fit(normed_train_data, train_labels,
                    epochs=EPOCHS, validation_split = 0.2, verbose=0,
                    callbacks=[early_stop, tfdocs.modeling.EpochDots()])

plotter.plot({'Early Stopping': early_history}, metric = "mae")
plt.ylim([0, 10000])
plt.ylabel('MAE [Total Count Confirmed]')

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} Total Count Confirmed".format(mae))

test_predictions = model.predict(normed_test_data).flatten()
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [Total Count Confirmed]')
plt.ylabel('Predictions [Total Count Confirmed]')
lims = [0, 25000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

plt.show()
