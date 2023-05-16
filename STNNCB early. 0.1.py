import re
import json
import numpy as np
import tensorflow as tf
import pandas as pd
import discord
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Attention, LayerNormalization
from tensorflow.keras.layers import Dropout, Bidirectional
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import MultiHeadAttention
from config import params

intents = discord.Intents.all()
bot = discord.Client(intents=intents)

# tf.config
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

# Set up GPU usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Assign the values from the 'params' dictionary to variables
embedding_dim = params["embedding_dim"]
transformer_dim = params["transformer_dim"]
transformer_dropout = params["transformer_dropout"]
lstm_units_1 = params["lstm_units_1"]
dropout_1 = params["dropout_1"]
lstm_units_2 = params["lstm_units_2"]
dense_units = params["dense_units"]
dropout_2 = params["dropout_2"]
optimizer = params["optimizer"]
loss = params["loss"]
epochs = params["epochs"]
batch_size = params["batch_size"]
l1 = params["l1"]
l2 = params["l2"]
output_activation = params["output_activation"]
lr = params["lr"]

def clean_text(text):
    # Remove special characters and replace with a space
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
    # Remove leading and trailing spaces
    text = text.strip()
    # Convert to lowercase
    text = text.lower()
    return text

load_pretrained = "n"  # Set the value to "n" to skip the prompt

if load_pretrained.lower() == "n":
    # Start training from scratch
    print("Starting training...")
    # Load the training data from JSON and extract inputs and outputs
    with open("output -copy.json", "r") as f:
        training_data = json.load(f)

        # Extract the user input and bot output columns as separate Pandas series
        user_input = [data['user_input'] for data in training_data]
        bot_output = [data['bot_output'] for data in training_data]

        # Split the data into training and test sets
        user_input_train, user_input_test, bot_output_train, bot_output_test = train_test_split(
            user_input, bot_output, test_size=0.2, random_state=42, shuffle=True
        )

        # Combine the training and test sets into separate conversation pairs
        train_data = [
            {"user_input": user_input_train[i], "bot_output": bot_output_train[i]}
            for i in range(len(user_input_train))
        ]

        test_data = [
            {"user_input": user_input_test[i], "bot_output": bot_output_test[i]}
            for i in range(len(user_input_test))
        ]

    # Combine the input and output for tokenization
    combined_text = user_input + bot_output

    # Create a tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(combined_text)

    # Convert text to sequences of indices
    user_input_sequences = tokenizer.texts_to_sequences(user_input)
    bot_output_sequences = tokenizer.texts_to_sequences(bot_output)

    # Get the maximum sequence length
    max_seq_length = max(max(map(len, user_input_sequences)), max(map(len, bot_output_sequences)))

    # Pad sequences to the maximum length
    user_input_sequences_padded = pad_sequences(user_input_sequences, maxlen=max_seq_length, padding='post')
    bot_output_sequences_padded = pad_sequences(bot_output_sequences, maxlen=max_seq_length, padding='post')

    # Convert sequences to numpy arrays
    user_input_sequences_padded = np.array(user_input_sequences_padded)
    bot_output_sequences_padded = np.array(bot_output_sequences_padded)

    # Print the tokenizer vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    print("Vocabulary size:", vocab_size)

    # Convert bot_output_sequences_padded to one-hot encoding
    bot_output_sequences_encoded = to_categorical(bot_output_sequences_padded, num_classes=vocab_size)

    # Verify the shape
    print("Shape of bot_output_sequences_encoded:", bot_output_sequences_encoded.shape)

# Define your optimizer with the learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=params["lr"])

# Use the variables in the model architecture and training
input_layer = tf.keras.Input(shape=(max_seq_length,))
embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim)(input_layer)
transformer_layer = tf.keras.layers.MultiHeadAttention(
    num_heads=4,
    key_dim=transformer_dim,
    dropout=transformer_dropout
)(embedding_layer, embedding_layer)
lstm_layer = tf.keras.layers.LSTM(lstm_units_2)(transformer_layer)
lstm_layer = tf.keras.layers.ActivityRegularization(l1=l1, l2=l2)(lstm_layer)
dense_layer = tf.keras.layers.Dense(dense_units, activation="relu")(lstm_layer)
dropout_layer2 = tf.keras.layers.Dropout(dropout_2)(dense_layer)
output_layer = tf.keras.layers.Dense(vocab_size, activation=output_activation)(dropout_layer2)

# Create the model
model = tf.keras.Model(input_layer, output_layer)

# Compile the model
model.compile(optimizer=optimizer, loss=params["loss"], metrics=['accuracy'])

print("Shape of user_input_sequences_padded:", user_input_sequences_padded.shape)
print("Shape of bot_output_sequences_padded:", bot_output_sequences_padded.shape)
print("max_seq_length:", max_seq_length)

# Convert bot_output_sequences_padded to categorical
bot_output_sequences_categorical = to_categorical(bot_output_sequences_padded, num_classes=vocab_size)

# Train-test split for padded sequences
user_input_train_padded, user_input_test_padded, bot_output_train_categorical, bot_output_test_categorical = train_test_split(
    user_input_sequences_padded, bot_output_sequences_categorical, test_size=0.2, random_state=42
)

# Split the data into training and test sets
user_input_train, user_input_test, bot_output_train, bot_output_test = train_test_split(
    user_input_sequences_padded,
    bot_output_sequences_categorical,
    test_size=0.2,
    random_state=42
)

# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    "STNNCB_model.h5",
    monitor="val_accuracy",  # Monitor validation accuracy
    save_best_only=True,  # Save only the best model
    mode="max"  # Select the maximum validation accuracy
)

# Define a function to calculate accuracy
def calculate_accuracy(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), dtype=tf.float32))
    return accuracy

# Train the model
history = model.fit(
    user_input_train,
    bot_output_train,
    validation_data=(user_input_test, bot_output_test),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[checkpoint_callback]
)

# Print validation accuracy at each epoch
for epoch, acc, val_acc in zip(range(len(history.history['accuracy'])), history.history['accuracy'], history.history['val_accuracy']):
    print(f"Epoch {epoch+1}/{epochs}: Accuracy = {acc:.4f}, Validation Accuracy = {val_acc:.4f}")

# Save the trained model
model.save("SNNCB_model.h5")

# Load the model
model = load_model('SNNCB_model.h5')

print(model.summary())

def handle_unknown_word(word_index):
    if word_index >= len(tokenizer.word_index):
        return "<unknown>"
    return tokenizer.index_word[word_index]

# Generate responses
tries = 0
max_tries = 10  # Maximum number of conversation tries
previous_inputs = []  # Track previous user inputs

@bot.event
async def on_message(message):
    global tries  # Declare 'tries' as a global variable

    if message.author == bot.user:
        return

    input_text = message.content  # Get user input from Discord message

    if not input_text:
        return  # Ignore empty input

    previous_inputs.append(input_text)

    input_seq = tokenizer.texts_to_sequences([input_text])
    if not input_seq:
        return  # Ignore input that cannot be tokenized

    min_response_length = len(input_text.split())

    input_seq = input_seq[0]
    input_padded = pad_sequences([input_seq], maxlen=max_seq_length, padding="post")
    response = ""
    response_len = 0

    while response_len < min_response_length:
        predicted_probs = model.predict(input_padded)[0]
        predicted_index = np.argmax(predicted_probs)
        predicted_word = handle_unknown_word(predicted_index + 1)

        if predicted_word == "<end>":
            break

        response += predicted_word + " "
        response_len += 1
        input_seq = np.append(input_seq, predicted_index)
        input_padded = pad_sequences([input_seq], maxlen=max_seq_length, padding="post")

    # Check the channel ID before sending the response
    if message.channel.id == 1086025739666739301:
        await message.channel.send(f"Bot: {response}")  # Send the response back to the same channel

    tries += 1

# Load bot token from config.json
with open('config.json', 'r') as f:
    config = json.load(f)
bot_token = config['bot_token']

bot.run(bot_token)