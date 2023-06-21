import numpy as np
import pandas as pd
from keras.layers import *
from keras.optimizers import *
import requests
from utils import WindowGenerator
from mqtt import MyMqtt
import paho.mqtt.publish as publish
from datetime import datetime, timezone,timedelta


url = 'http://209.38.208.230:8000/api/posts/?date_range=year&dev='

broker = "159.89.103.242"  # Replace with your MQTT broker address
port = 1883  # Replace with the appropriate port
client_id = "your_client_id_999"  # Replace with your desired client ID
mqtt_client = MyMqtt(broker, port, client_id)
mqtt_client.connect()
mqtt_client.subscribe("tensor/#")  # Replace with your desired topic

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
#mqtt_client.publish("forecast", "00")
def handle_message(topic, payload):
    
    range = topic.split("/")[1]
    dev = payload[2:-1]
    if dev and range:  
        response=requests.get(url+dev).json()
        df1=pd.DataFrame(response)

        series = df1[['created', 'value']]

        series.drop_duplicates(subset='created', inplace=True)
        series['created'] = pd.to_datetime(series['created'])

        date_time = pd.to_datetime(series.pop('created'), format='%d.%m.%Y %H:%M:%S')

        timestamp_s = date_time.map(pd.Timestamp.timestamp)

        day = 24*60*60
        #year = (365.2425)*day

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
        
        wide_window = WindowGenerator(
        input_width=24, label_width=24, shift=1, train_df = train_df, val_df = val_df, test_df = test_df,
        label_columns=['value'])
        
        history = compile_and_fit(lstm_model, wide_window)

        prediction_data = wide_window.make_dataset(test_df)
        predictions = lstm_model.predict(prediction_data)
        predictions_original = (predictions * train_std['value']) + train_mean['value']
        
        if range == 'today':            
            pub_topic = "forecast/"+dev+'/'+range
            current_datetime = datetime.now() - timedelta(days=1)
            # Get tomorrow's date
            tomorrow = current_datetime + timedelta(days=1)
            # Set the starting time to the beginning of tomorrow
            starting_time = datetime(tomorrow.year, tomorrow.month, tomorrow.day)
            next_day = starting_time + timedelta(days=1)
            for val in predictions_original[0]:
                date_string = starting_time.strftime("%Y-%m-%d %H:%M:%S")
                hour_obj = {
                    "date":date_string,
                    "power": val[0]
                }
                starting_time += timedelta(hours=1)
                
                mqtt_client.publish(pub_topic, str(hour_obj))

    
mqtt_client.callback = handle_message
    
mqtt_client.start()






