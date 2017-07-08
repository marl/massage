import guitar_strummer

# please change line 4-8
jams_file = '/Users/tom/Dropbox/piano_guitar_resynth/chord_predictions/acoustic/AHa_TakeOnMe_STEM_04.jams'
audio_path = '/Users/tom/Dropbox/piano_guitar_resynth/tom_guitar/acoustic/AHa_TakeOnMe_STEM_04.wav'
instrument = 'acoustic guitar'
output_path = '/Users/tom/Dropbox/piano_guitar_resynth/tom-exp/test_repo.wav'
output_fs = 44100

y_stereo, midi_data = guitar_strummer.resynth_guitar_rabitt(jams_file, audio_path, instrument, output_fs)
guitar_strummer.write_small_wav(output_path, y_stereo)
