import keras
import numpy as np

from encode import encode, SEQUENCE_LENGTH
from parse import SAVE_DIR


def generate_training_sequences(sequence_length):
    
    # load the encoded songs
    int_songs = encode(dataset_path=SAVE_DIR)
    
    # generate the training sequences
    inputs = []
    targets = []
    
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])
    
    # one-hot encode the sequences
    # the shape of the input: (num_sequences, sequence_length, vocabulary_size)
    # one-hot encoding example: [ [0,1,2], [1,1,2] ] => [ [[1,0,0], [0,1,0], [0,0,1]], [[0,1,0], [0,1,0], [0,0,1]] ]
    vocabulary_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    return inputs, targets

if __name__ == "__main__":
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)