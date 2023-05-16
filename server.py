from flask import Flask, jsonify, request
from flask_cors import CORS
import subprocess
import socket

app = Flask(__name__)
CORS(app)

# Initial configuration parameters
params = {
    "embedding_dim": 1024,
    "transformer_dim": 1024,
    "transformer_dropout": 0.2,
    "lstm_units_1": 512,
    "dropout_1": 0.3,
    "lstm_units_2": 256,
    "dense_units": 1463,
    "dropout_2": 0.3,
    "optimizer": "relu",
    "loss": "categorical_crossentropy",
    "epochs": 1000,
    "batch_size": 64,
    "l1": 0.01,
    "l2": 0.01,
    "output_activation": "softmax",
    "lr": 0.003
}

# Default configuration parameters
default_config = {}


def load_default_config():
    global default_config
    try:
        with open("default_config.py", "r") as config_file:
            config_code = compile(config_file.read(), "default_config.py", 'exec')
            exec(config_code, default_config)
    except FileNotFoundError:
        print("Default configuration file 'default_config.py' not found.")


def generate_python_config_string(config_dict):
    config_string = "params = {\n"
    for key, value in config_dict.items():
        config_string += f"    '{key}': {repr(value)},\n"
    config_string += "}"
    return config_string


def update_config_file():
    config_string = generate_python_config_string(params)
    with open("config.py", "w") as config_file:
        config_file.write(config_string)


def start_chatbot_process():
    # Change the path and command accordingly
    with open("chatbot_output.txt", "w") as output_file:
        subprocess.Popen(["python", "STNNCB early. 0.1.py"], stdout=output_file)


# DNS resolution function using Google DNS (8.8.8.8)
def resolve_dns(hostname):
    try:
        resolver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        resolver.connect(("8.8.8.8", 53))
        resolver.sendall(b"\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00"
                         + hostname.encode("utf-8") + b"\x00\x00\x01\x00\x01")
        data = resolver.recv(1024)
        resolver.close()
        ip_address = socket.inet_ntoa(data[21:25])
        return ip_address
    except socket.error:
        return None


# Endpoint to retrieve the current configuration
@app.route('/get_config', methods=['GET'])
def get_config():
    return jsonify(params)


# Endpoint to update the configuration
@app.route('/update_config', methods=['POST'])
def update_config_endpoint():
    new_config = request.get_json()
    params.update(new_config)
    update_config_file()
    return jsonify(success=True)


# Endpoint to send a message and receive a response
@app.route('/send_message', methods=['POST'])
def send_message():
    message = request.json['message']
    # Process the message and generate a response
    response = process_message(message)  # Replace with your message processing logic
    return jsonify({'response': response})


# Endpoint to retrieve messages from chatbot_output.txt
@app.route('/get_chatbot_output', methods=['GET'])
def get_chatbot_output():
    with open("chatbot_output.txt", "r") as output_file:
        chatbot_output = output_file.read()
    return jsonify({'chatbot_output': chatbot_output})


# DNS resolution function using Google DNS (8.8.8.8)
def resolve_dns(hostname):
    try:
        resolver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        resolver.connect(("8.8.8.8", 53))
        resolver.sendall(b"\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00"
                         + hostname.encode("utf-8") + b"\x00\x00\x01\x00\x01")
        data = resolver.recv(1024)
        resolver.close()
        ip_address = socket.inet_ntoa(data[21:25])
        return ip_address
    except socket.error:
        return None


# Endpoint to start the chatbot
@app.route('/start_chatbot', methods=['GET'])
def start_chatbot():
    start_chatbot_process()
    return jsonify(success=True)


# Endpoint to configure the settings
@app.route('/configure_settings', methods=['POST'])
def configure_settings():
    new_settings = request.get_json()
    params.update(new_settings)
    update_config_file()
    return jsonify(success=True)


if __name__ == '__main__':
    app.run(host=resolve_dns("your-domain.com"), port=11000)