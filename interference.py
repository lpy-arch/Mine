# pylint: disable=E1102

import json
import music21 as m21
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from encode import SEQUENCE_LENGTH, MAPPING_PATH
from torch_LSTM import SAVE_MODEL_PATH, LSTM

NUM_STEPS = 100
TEMPERATURE = 0.5

class MelodyGenerator:
    def __init__(self, model, model_path=SAVE_MODEL_PATH):
        
        self.model_path = model_path
        self.model = model
        
        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)
        
        self._start_symbols = ["/"] * SEQUENCE_LENGTH
        
    def _sample_with_temperature(self, probabilities, temperature):
        
        predictions = np.log(probabilities + 1e-10) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))
        
        choices = range(len(probabilities))    # [0,1,2,3]
        index = np.random.choice(choices, p=probabilities)
        
        return index
        
        
    def generate_melody(self, seed, num_steps, max_seed_length, temperature):
        
        # create seed with start symbols
        seed = seed.split()
        melody = seed
        # we only will use the last max_seed_length after
        # so for this step we just want to make sure that before the seed, we have enough number of / to input sequence_length number of code
        seed = self._start_symbols + seed
        
        # map seed to integers
        seed = [self._mappings[symbol] for symbol in seed]
        
        # 
        for _ in range(num_steps):
            
            # limit the seed to the max_seed_length
            seed = seed[-max_seed_length:]
            
            # one-hot encode the seed
            # (1, max_seed_length, num of symbols in the vocabulary)
            seed = torch.tensor(seed)
            onehot_seed = F.one_hot(seed, num_classes=38)
            onehot_seed = onehot_seed.unsqueeze(0)
            onehot_seed = onehot_seed.float()
            
            # make a prediction
            # [0.1, 0.2, 0.1, 0.6]
            probabilities = self.model(onehot_seed)
            probabilities = nn.Softmax(dim=-1)(probabilities)
            probabilities = probabilities[0]
            # print(probabilities)

            probabilities = probabilities.detach().numpy()
            output_int = self._sample_with_temperature(probabilities, temperature)
            
            # update seed
            seed = seed.tolist()
            seed.append(output_int)
            
            # map int to our encoding
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]
            
            # check whether we are at the end of a melody
            if output_symbol == "/":
                break
            
            # update the melody
            melody.append(output_symbol)
            
        return melody
    
    
    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="mel.mid"):
        
        # create a music21 stream
        stream = m21.stream.Stream()
        
        # parse all the symbol in the melody and create note/rest objects
        # 60 _ _ _ r _ 62 _
        start_symbol = None
        step_counter = 1
        
        for i, symbol in enumerate(melody):
            
            # handle case in which we have a note/rest
            if symbol != "_" or i+1 == len(melody):
                
                # ensure we are not dealing with the initial note/rest
                if start_symbol is not None:
                    
                    quarter_length_duration = step_duration * step_counter
                    
                    # handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                    
                    # handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)
                        
                    stream.append(m21_event)
                    
                    # reset the step counter
                    step_counter = 1
                    
                # update the start_symbol
                start_symbol = symbol
            
            # handle case in which we have a prolongation sign "_"
            else:
                step_counter += 1
                
        # write the m21 stream to MIDI file
        stream.write(format, file_name)

if __name__ == "__main__":
    model = LSTM()
    model.load_state_dict(torch.load(SAVE_MODEL_PATH))
    model.eval()

    mg = MelodyGenerator(model)
    seed = "67 _ _ _ _ _ 65 _ 64 _ 62 _ 60 _ _ _"
    melody = mg.generate_melody(seed, NUM_STEPS, SEQUENCE_LENGTH, TEMPERATURE)
    print(melody)
    # mg.save_melody(melody)