import os
import json
import keras
import music21 as m21
import numpy as np

KERN_DATASET_PATH = "deutschl/erk"
ACCEPTABLE_DURATIONS = [
    0.25,
    0.5,
    0.75,
    1,
    1.5,
    2,
    3,
    4
]
SAVE_DIR = "dataset"
SINGLE_FILE_DATASET = "file_dataset"
WINDOW_LENGTH = 64
MAPPING_PATH = "mapping.json"

def load_songs_in_kern(dataset_path):
    
    songs = []
    
    # go through all the file in dataset and load the with music21
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs     


def has_acceptable_durations(song, acceptable_durations):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True


def transpose(song):
    
    # get key from the song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]
    
    # estimate key using music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")
    
    # get the interval for transposition
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))
    
    # transpose song by calculated interval
    transposed_song = song.transpose(interval)
    
    return transposed_song


def encode_song(song, time_step=0.25):
    
    # p = 60, d = 1.0 -> [60, "_", "_", "_"]
    
    encoded_song = []
    
    # Analyse MIDI with the "event" way, means that only at the begining of the event,
    # there will be a note, and for the rest of the event, it just last for an amount of time
    for event in song.flat.notesAndRests:
        
        # handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi  # 60
        # handle rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"
            
        # convert the note/rest into time series notation
        # every symbol represents 16th note
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")
                
    # cast encoded song to a str
    encoded_song = " ".join(map(str, encoded_song))
    
    return encoded_song



def create_single_file_dataset(dataset_path, file_dataset_path, window_length):
    new_song_delimiter = "/ " * window_length
    songs = ""
    
    # load encoded songs and add delimiters
    # 如果出现隐藏名称干扰文件读取，就在数据集文件夹下运行“find . -name '.DS_Store' -type f -delete”
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter
    songs = songs[:-1]    # delete the " " in the end
    
    # save string that contains all dataset
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)
        
    return songs


def creat_mapping(songs, mapping_path):
    mappings = {}
    
    # identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))
    
    # create mappings
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i
    
    # save vocabulary to json file
    with open(mapping_path, "w", encoding='utf-8') as fp:
        json.dump(mappings, fp, indent=4)


def convert_songs_to_int(songs):
    
    # Use the Mapping to convert songs into integers
    int_songs = []
    
    # load the mappings
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)
    
    # cast song strings to strings
    songs = songs.split()
    
    # map songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])
        
    return int_songs


def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song


def preprocess(dataset_path):
    
    # load the folk songs
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")
    
    for i, song in enumerate(songs):
        
        # filter out songs that have non-acceptable durations 
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue

        # transport songs to Cmaj/Amin
        song = transpose(song)
        
        # encode songs with music time series representations
        encoded_song = encode_song(song)
        

        # save songs to text file
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)


def generate_training_sequences(window_length):
    
    # load the songs and map them to int
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)
    
    # generate the training sequences
    inputs = []
    targets = []
    
    num_sequences = len(int_songs) - window_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+window_length])
        targets.append(int_songs[i+window_length])
    
    # one-hot encode the sequences
    # the shape of the input: (num_sequences, window_length, vocabulary_size)
    # one-hot encoding example: [ [0,1,2], [1,1,2] ] => [ [[1,0,0], [0,1,0], [0,0,1]], [[0,1,0], [0,1,0], [0,0,1]] ]
    vocabulary_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    return inputs, targets

if __name__ == "__main__":
    # the whole preprocess
    preprocess(KERN_DATASET_PATH)

    # single file collection
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, WINDOW_LENGTH)

    # create vocabulary map
    creat_mapping(songs, MAPPING_PATH)

    # training sequences generation
    inputs, targets = generate_training_sequences(WINDOW_LENGTH)