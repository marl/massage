import os
import numpy as np

SF_PATH = os.path.join(os.path.dirname(__file__), '../resources/sf2')
ACOUSTIC_MFCC_NPZ_PATH = os.path.join(os.path.dirname(__file__), '../resources/mfcc_npz/acoustic_sf_mfcc.npz')
ACOUSTIC_SF_MFCC = np.load(ACOUSTIC_MFCC_NPZ_PATH)
VOICING_FILE = os.path.join(os.path.dirname(__file__), '../resources/chord_voicings.json')