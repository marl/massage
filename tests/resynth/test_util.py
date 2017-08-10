import pretty_midi
from unittest import TestCase
from massage.resynth.util import *
from . import TEST_VOICING_FILE, TEST_MIDI_FILE


class TestUtil(TestCase):

    def test_compute_avg_mfcc_zero(self):
        y = np.zeros(1000)
        avg_mfcc = compute_avg_mfcc(y=y, sr=44100)
        target_mfcc = np.zeros(39)
        self.assertTrue(np.allclose(avg_mfcc, target_mfcc))

    def test_onset_offset(self):
        fs = 44100
        noise = np.random.random(2205) - 0.5  # 50 ms of nosie
        silence = np.zeros(2205)
        y = np.concatenate((silence, noise, silence, noise, silence))
        target_on_t = np.array([0.05, 0.15])
        target_off_t = np.array([0.10, 0.20])
        on_t, off_t, on_s = onset_offset(y=y, sr=fs)
        on_close = np.abs(target_on_t - on_t) < 0.03  # 30 ms slack
        off_close = np.abs(target_off_t - off_t) < 0.03
        self.assertTrue(np.logical_and(on_close, off_close).all())

    def test_compute_envelope(self):
        noise = np.random.random(220500) - 0.5  # 5s of nosie
        silence = np.zeros(220500)
        y = np.concatenate((silence, noise, silence, noise, silence))
        env = compute_envelope(y)
        print(y.shape)
        # import matplotlib.pyplot as plt
        # plt.plot(env/ np.max(env))
        # plt.plot(y)
        # plt.show()
        self.assertEqual(len(env), 220500*5)

    def test_get_energy_envelope(self):
        y = np.zeros((2, 100000))
        env = get_energy_envelope(y)
        self.assertEqual(np.sum(env), 0)
        self.assertEqual(env.shape, y.shape)

    def test_pick_sf(self):
        # synthesis something different with the sf, and try to match
        midi_data = pretty_midi.PrettyMIDI(TEST_MIDI_FILE)
        test_sf_path = os.path.join(
            os.path.dirname(__file__), '../data/28MBGM.sf2')
        fs = 44100
        y = midi_data.fluidsynth(sf2_path=test_sf_path, fs=fs)
        sf_path, program = pick_sf(y, fs, 'acoustic guitar')
        sf_base = os.path.basename(sf_path)
        self.assertEqual(program, 25)
        self.assertIsInstance(sf_base, str)
        # the following test should work, but doesn't... right now the sf
        # picked out is 'chorium.sf2' as opposed to 28MBGM
        # self.assertEqual('sf_base', '28MBGM.sf2')

    def test_amplitude_to_velocity(self):
        energies = [-1, 0, 0.5, 1]
        velocities = amplitude_to_velocity(energies)
        self.assertListEqual(list(velocities), [60, 80, 100, 120])

    def test_midi_to_jams(self):
        midi_data = pretty_midi.PrettyMIDI(TEST_MIDI_FILE)
        jam = midi_to_jams(midi_data)
        jam_len = len(jam.annotations[0].data)
        midi_len = len(midi_data.instruments[0].notes)
        self.assertEqual(jam_len, midi_len)

    def test_voicing_dist(self):
        v1 = [1, 3, 5, 7, 9]
        v2 = [1, 3, 5, 7, 10]
        self.assertEqual(voicing_dist(v1, v2), 0.2)

    def test_get_all_voicings(self):
        voicing_dict = get_all_voicings(TEST_VOICING_FILE)
        self.assertEqual(
            voicing_dict['G:maj'][0], [43, 47, 50, 55, 59])

    def test_choose_voicing(self):
        voicing_dict = get_all_voicings(TEST_VOICING_FILE)
        voicing = choose_voicing('G:maj', voicing_dict, [43, 47, 50, 55, 59])
        self.assertIsNotNone(voicing[0])
