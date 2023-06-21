import paho.mqtt.client as mqtt

class MyMqtt:
    def __init__(self, broker, port, client_id, callback=None):
            self.client = mqtt.Client(client_id)
            self.client.on_connect = self.on_connect
            self.client.on_message = self.on_message
            self.broker = broker
            self.port = port
            self.connected = False  # Flag to track connection status
            self.callback = callback
            

    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code " + str(rc))
        # Subscribe to topics here if needed

    def on_message(self, client, userdata, msg):
        topic = msg.topic
        payload = str(msg.payload)
        if self.callback:
            self.callback(topic, payload)
        

    def connect(self):
        self.client.connect(self.broker, self.port, 60)
        self.connected = True

    def subscribe(self, topic):
        self.client.subscribe(topic)

    def publish(self, topic, message):
        self.client.publish(topic, message)

    def start(self):
        self.client.loop_forever()