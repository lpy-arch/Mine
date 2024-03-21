import os
import json
import music21 as m21

from parse import parse
from parse import SAVE_DIR

SINGLE_FILE_DATASET = "mid_data/file_dataset"
SEQUENCE_LENGTH = 64
MAPPING_PATH = "mid_data/mapping.json"


def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song

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
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")
                
    # cast encoded song to a str
    encoded_song = " ".join(map(str, encoded_song))
    
    return encoded_song

def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    new_song_delimiter = "/ " * sequence_length
    songs = ""
    
    # load encoded songs and add delimiters
    # 如果出现隐藏名称干扰文件读取，就在数据集文件夹下运行“find . -name '.DS_Store' -type f -delete”
    for path, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.DS_Store'):
                # 跳过 .DS_Store 类型的文件
                continue
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

def encode(dataset_path):

    # load the parsed songs
    songs = parse(dataset_path)

    for i, song in enumerate(songs):
        # encode songs with music time series representations
        encoded_song = encode_song(song)

        # save songs to text file
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)

    # single file collection
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)

    # create vocabulary map
    creat_mapping(songs, MAPPING_PATH)

    # load the songs and map them to int
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)

    return int_songs