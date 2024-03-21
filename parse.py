import os
import music21 as m21

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


def parse(dataset_path):
    
    parsed_songs = []
    # load the folk songs
    songs = load_songs_in_kern(dataset_path)

    for i, song in enumerate(songs):
        # filter out songs that have non-acceptable durations 
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue
    
        # transport songs to Cmaj/Amin
        song = transpose(song)

        # collect transposed songs
        parsed_songs.append(song)

    return parsed_songs