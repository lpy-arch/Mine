from preprocess import generate_training_sequences, WINDOW_LENGTH
from model import build_model

OUTPUT_UNITS = 38
NUM_UNITS = [256]
DENSE_UNITS = [38]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model.h5"

def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, dense_units=DENSE_UNITS, loss=LOSS, learning_rate=LEARNING_RATE):

    # generate the training sequences
    inputs, targets = generate_training_sequences(WINDOW_LENGTH) 
    
    # build the network
    model = build_model(output_units, num_units, dense_units, loss, learning_rate)
    
    # train the model
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # save the model
    model.save(SAVE_MODEL_PATH)

if __name__ == "__main__":
    train()