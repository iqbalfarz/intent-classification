import argparse
from transformers import pipeline
from typing import Union, Dict
import numpy as np
import librosa

SAMPLE_RATE = 16000 # model expects the audio to be in the 16K sample_rate
hub_model_id = "MuhammadIqbalBazmi/wav2vec2-base-intent-classification-ori"


def intent_inference(audio: Union[str, np.ndarray])->Dict:
    """
        this method takes an audio (filepath or np.ndarray) 
        and return the inference result in json/dict format with top-5 labels with the score

        Parameters
        ----------
        audio: Union[str, np.ndarray]
            Audio filepath or Audio in np.ndarray format with 16K sampling rate
        
        Returns
        -------
        Dictionary containing top-5 labels with their score in the below format
            ```
            [
                {'score': 0.9726924896240234, 'label': 'casual_talk_greeting'}, 
                {'score': 0.010468798689544201, 'label': 'casual_talk_goodbye'}, 
                {'score': 0.0033398999366909266, 'label': 'bike_modes'}, 
                {'score': 0.0031408704817295074, 'label': 'Locate_Dealer'}, 
                {'score': 0.0026542956475168467, 'label': 'About_iQube'}
            ]
            ```
    """
    # loading best wav2vec2 model
    model = pipeline("audio-classification", model=hub_model_id) 
    if isinstance(audio, str):
        audio, _ = librosa.load(audio, sr=SAMPLE_RATE)
    return model(audio)
 

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        "--audio",
                        help="input audio filepath",
                        required=True,
                       )
    args = parser.parse_args()
    print(intent_inference(args.audio))
