import numpy as np
import pandas as pd
from keras.layers import *
from keras.optimizers import *
import requests
from utils import WindowGenerator
# from mqtt import MyMqtt
import paho.mqtt.publish as publish
from paho.mqtt import client
#import paho.mqtt.client as mqtt
from datetime import datetime, timezone, timedelta
import traceback  # Import the traceback module for detailed error information
from sklearn.metrics import mean_absolute_error
import logging
import json
import joblib

# logging.basicConfig(level=logging.DEBUG)

url = 'http://209.38.208.230:8000/api/posts/?date_range=year&dev='

broker = "159.89.103.242"  # Replace with your MQTT broker address
port = 1883  # Replace with the appropriate port
client_id = "your_client_id_9991981"  # Replace with your desired client ID
keep_alive_interval = 60  # Set the Keep-Alive interval in seconds

def on_connect(client, userdata, flags, rc):
    client.subscribe("tensor/#")
    client.subscribe("predict/single")


MAX_EPOCHS = 50

def compile_and_fit(model, window, patience=2):
    try:
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
    except Exception as e:
        print("Error compiling and fitting model:", str(e))
        traceback.print_exc()  # Print detailed error information
        return None  # Return None to indicate failure

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

def validateJSON(jsonData):
    try:
        json.loads(jsonData)
    except ValueError as err:
        return False
    return True  

def make_predict(df,home):
    scaler = joblib.load("fat_scaler.pkl")
    single_data_scaled = scaler.transform(df)
    model = tf.keras.models.load_model('fat_model.h5')
    predicted_probability = model.predict(single_data_scaled)
    predicted_class = (predicted_probability >= 0.5).astype(int)
    prediction = predicted_class.item()
    
    if prediction == 1:
        pub_topic = "predict/result"
        home_obj = {
            "home":home
        }
        message = json.dumps(home_obj)
        client.publish(pub_topic, message)

def on_message(client, userdata, msg):
    try:
        topic = msg.topic
        if topic == "predict/single":
            is_valid = validateJSON(msg.payload)
            if is_valid:
                data_out=json.loads(msg.payload.decode())
                
                df_single = pd.DataFrame([data_out])                
                home = df_single['home'].to_string(index=False, header=False)
                #print(home)
                df_single_fin = df_single.drop('home', axis=1)
                make_predict(df_single_fin, home)       
                
                
                
        else:        
            range = topic.split("/")[1]
            dev = msg.payload.decode('utf-8')     
            if dev and range:  
                response=requests.get(url+dev).json()
                df1=pd.DataFrame(response)
                
                if len(df1) >= 100:        
                    series = df1[['created', 'value']]

                    series.drop_duplicates(subset='created', inplace=True)
                    series['created'] = pd.to_datetime(series['created'])

                    date_time = pd.to_datetime(series.pop('created'), format='%d.%m.%Y %H:%M:%S')

                    timestamp_s = date_time.map(pd.Timestamp.timestamp)

                    day = 24*60*60
                    week = day*7

                    series['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
                    series['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
                    series['Week sin'] = np.sin(timestamp_s * (2 * np.pi / week))
                    series['Week cos'] = np.cos(timestamp_s * (2 * np.pi / week))
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
                    print("HERE!!!!!")
                    history = compile_and_fit(lstm_model, wide_window)

                    if history is not None:
                        prediction_data = wide_window.make_dataset(test_df)
                        predictions = lstm_model.predict(prediction_data)
                        predictions_original = (predictions * train_std['value']) + train_mean['value']
                        final_loss = history.history['loss'][-1]
                        formatted_final_loss = "{:.2f}".format(final_loss)
                        final_mae = history.history['mean_absolute_error'][-1]
                        formatted_final_mae = "{:.2f}".format(final_mae)
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
                                    "power": val[0],
                                    "loss": formatted_final_loss,
                                    "mae": formatted_final_mae
                                }
                                print(hour_obj)
                                starting_time += timedelta(hours=1)                            
                                client.publish(pub_topic, str(hour_obj))
    except Exception as e:
        print("Error handling message:", str(e))
        traceback.print_exc()  # Print detailed error information

client = client.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("159.89.103.242", 1883)
#client.connect_async("159.89.103.242", 1883)

client.loop_forever()