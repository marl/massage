"""
Test for massage.resynth.guitar_strummer
"""

import os
from unittest import TestCase
import mock

import jams
import librosa
import numpy as np

from massage.resynth import guitar_strummer
import massage.resynth.util as util
import pretty_midi

import sys
sys.modules['fluidsynth'] = mock.MagicMock()

TEST_VOICING_FILE = os.path.join(
    os.path.dirname(__file__), '../data/chord_voicings.json')
TEST_RUN_MOCK_Y = os.path.join(
    os.path.dirname(__file__), '../data/test_run_mock_y.npz')


class TestGuitarStrummer(TestCase):
    def setUp(self):
        self.strummer = guitar_strummer.GuitarStrummer()

    @mock.patch.object(pretty_midi.PrettyMIDI, 'fluidsynth', autospec=True)
    def test_run(self, mock_fluidsynth):
        jams_file = os.path.join(
            os.path.dirname(__file__), '../data/acoustic_guitar.jams')
        audio_path = os.path.join(
            os.path.dirname(__file__), '../data/acoustic_guitar.wav')
        instrument_label = 'acoustic guitar'

        mock_y = np.load(TEST_RUN_MOCK_Y)['arr_0']
        mock_fluidsynth.return_value = mock_y

        y, fs = librosa.load(audio_path, sr=None, mono=False)
        jam = jams.load(jams_file)
        y_stereo, jams_out = self.strummer.run(y, fs, jam, instrument_label)
        # Check y.shape, check y is not all zeros
        self.assertEqual(y_stereo.shape[0], 2)
        self.assertNotEqual(np.max(np.abs(y_stereo)), 0)
        # Check jams have the right type(s) of annotation
        anns = jams_out.annotations
        self.assertEqual(
            [anns[i]['namespace'] for i in range(len(anns))], ['pitch_midi'])

    def test_run_empty_y(self):
        try:
            self.strummer.run(y=np.array([]), fs=0)
        except ValueError:
            pass

    def test_run_empty_jams(self):
        mock_jams = jams.JAMS()
        mock_y = np.zeros(10000)
        try:
            self.strummer.run(mock_y, 44100, mock_jams)
        except ValueError:
            pass

    def test_get_id(self):
        self.assertEqual(self.strummer.get_id(), 'guitar_strummer')

    def test__get_strum_1(self):
        start_t = 0.5
        end_t = 1
        chord = 'Ab:maj6'
        voicings = util.get_all_voicings(TEST_VOICING_FILE)
        prev_voicing = [43, 47, 50, 55, 59]
        backwards = True
        velocity = 30

        strum, current_voicing = self.strummer._get_strum(
            start_t, end_t, chord, voicings, prev_voicing, backwards, velocity)

        self.assertIsInstance(strum[0], pretty_midi.Note)
        self.assertEqual(len(strum), len(current_voicing))

    def test__get_strum_2(self):
        start_t = 0
        end_t = 1
        chord = 'Ab:maj6'
        voicings = util.get_all_voicings(TEST_VOICING_FILE)
        prev_voicing = [43, 47, 50, 55, 59]
        backwards = False
        velocity = 127

        strum, current_voicing = self.strummer._get_strum(
            start_t, end_t, chord, voicings, prev_voicing, backwards, velocity)

        self.assertIsInstance(strum[0], pretty_midi.Note)
        self.assertEqual(len(strum), len(current_voicing))

    def test__get_strum_3(self):
        start_t = 1
        end_t = 5
        chord = 'Ab:maj6'
        voicings = util.get_all_voicings(TEST_VOICING_FILE)
        prev_voicing = [43, 47, 50, 55, 59]
        backwards = False
        velocity = None

        strum, current_voicing = self.strummer._get_strum(
            start_t, end_t, chord, voicings, prev_voicing, backwards, velocity)

        self.assertIsInstance(strum[0], pretty_midi.Note)
        self.assertEqual(len(strum), len(current_voicing))

    def test__get_strum_4(self):
        start_t = 1
        end_t = 1.00000001
        chord = 'Ab:maj6'
        voicings = util.get_all_voicings(TEST_VOICING_FILE)
        prev_voicing = [43, 47, 50, 55, 59]
        backwards = False
        velocity = None

        passed = False
        try:
            strum, current_voicing = self.strummer._get_strum(
                start_t, end_t, chord, voicings, prev_voicing, backwards,
                velocity
            )
        except ValueError:
            passed=True

        self.assertTrue(passed)

    def test__generate_chord_midi(self):
        chord_seq = [[0.0, 0.9, 'Ab:maj6'], [1.0, 2.0, 'G:maj']]
        onsets = np.array([0.0, 1.0])
        offsets = np.array([0.9, 2.0])
        velocities = np.array([90, 80])

        def interp(x):
            return 0.9

        interps = [interp, interp]
        midi = self.strummer._generate_chord_midi(
            chord_seq, onsets, offsets, velocities, 25, interps,
            TEST_VOICING_FILE)
        chords = midi.instruments[0].notes
        self.assertGreaterEqual(len(chords), 8)
        # midi.write('test_midi.mid')
        self.assertIsInstance(midi, pretty_midi.PrettyMIDI)

    def test__generate_chord_midi_2(self):
        chord_seq = [[0.0, 0.9, 'N'], [1.0, 2.0, 'G:maj']]
        onsets = np.array([0.0, 1.0])
        offsets = np.array([0.9, 2.0])
        velocities = np.array([90, 80])

        def interp(x):
            return 0.1

        interps = [interp, interp]
        midi = self.strummer._generate_chord_midi(
            chord_seq, onsets, offsets, velocities, 25, interps,
            TEST_VOICING_FILE, energy_thresh=0.3)
        chords = midi.instruments[0].notes
        self.assertIsInstance(midi, pretty_midi.PrettyMIDI)

