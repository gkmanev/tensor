import numpy as np
import pandas as pd
from keras.layers import *
from keras.optimizers import *
import requests
from utils import WindowGenerator

url = 'http://209.38.208.230:8000/api/posts/?date_range=year&dev=sm-0006'
response=requests.get(url).json()
df1=pd.DataFrame(response)

series = df1[['created', 'value']]

series.drop_duplicates(subset='created', inplace=True)
series['created'] = pd.to_datetime(series['created'])

date_time = pd.to_datetime(series.pop('created'), format='%d.%m.%Y %H:%M:%S')

timestamp_s = date_time.map(pd.Timestamp.timestamp)

day = 24*60*60
year = (365.2425)*day

series['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
series['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
# series['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
# series['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

column_indices = {name: i for i, name in enumerate(series.columns)}

n = len(series)
train_df = series[0:int(n*0.7)]
val_df = series[int(n*0.7):int(n*0.9)]
test_df = series[int(n*0.9):]

num_features = series.shape[1]

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std

val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std


df_std = (series - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')


MAX_EPOCHS = 50

def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history


lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=1, train_df = train_df, val_df = val_df, test_df = test_df,
    label_columns=['value'])

history = compile_and_fit(lstm_model, wide_window)

prediction_data = wide_window.make_dataset(test_df)
predictions = lstm_model.predict(prediction_data)
predictions_original = (predictions * train_std['value']) + train_mean['value']

print(predictions_original[0])