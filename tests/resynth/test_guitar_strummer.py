"""
Test for massage.resynth.guitar_strummer
"""

import unittest
import jams
import librosa
import os
from massage.resynth import guitar_strummer

def relpath(f):
    return os.path.join(os.path.dirname(__file__), f)

jams_file = relpath('../data/acoustic_guitar.jams')
audio_path = relpath('../data/acoustic_guitar.wav')
instrument_label = relpath('acoustic guitar')

output_path = '/Users/tom/Desktop/acoustic_guitar_test.wav'

def write_small_wav(save_path, y, fs=44100, bitdepth=16):
    fhandle, tmp_file = tempfile.mkstemp(suffix='.wav')
    librosa.output.write_wav(tmp_file, y, fs)
    tfm = sox.Transformer()
    tfm.convert(bitdepth=bitdepth)
    tfm.build(tmp_file, save_path)
    os.close(fhandle)
    os.remove(tmp_file)

class TestGuitarStrummer(unittest.TestCase):

    def setUp(self):
        self.syn = guitar_strummer.GuitarStrummer()

    def test_run(self):
        y, fs = librosa.load(audio_path, sr = None, mono = False)
        jam = jams.load(jams_file)
        y_stereo, jams_out = self.syn.run(y, fs, jam, instrument_label)
        # Check y.shape, check y is not all zeros, check jams_out has right annotation namefield
        self.assertEqual(y_stereo.shape[0], 2)