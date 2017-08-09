import jams
import os
import librosa
import sox
import tempfile

from massage.resynth import guitar_strummer

def write_small_wav(save_path, y, fs=44100, bitdepth=16):
    fhandle, tmp_file = tempfile.mkstemp(suffix='.wav')
    librosa.output.write_wav(tmp_file, y, fs)
    tfm = sox.Transformer()
    tfm.convert(bitdepth=bitdepth)
    tfm.build(tmp_file, save_path)
    os.close(fhandle)
    os.remove(tmp_file)

# please change line 4-8
jams_file = '../../tests/data/acoustic_guitar.jams'
audio_path = '../../tests/data/acoustic_guitar.wav'
instrument_label = 'acoustic guitar'
output_path = '/Users/tom/Desktop/acoustic_guitar_test.wav'


syn = guitar_strummer.GuitarStrummer()

y, fs = librosa.load(audio_path, sr = None, mono = False)
jam = jams.load(jams_file)
y_stereo, jams_out = syn.run(y, fs, jam, instrument_label)
write_small_wav(output_path, y_stereo)