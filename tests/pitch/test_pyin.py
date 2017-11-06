"""Test for massage.pitch.pyin
"""
import unittest
import os
import numpy as np

from massage.pitch import pyin


def relpath(f):
    return os.path.join(os.path.dirname(__file__), f)


AUDIO_FILE = relpath("../data/vocal.wav")


def array_equal(array1, array2):
    return np.all(np.isclose(array1, array2))


class TestPyin(unittest.TestCase):

    def setUp(self):
        self.ptr = pyin.Pyin()

    def test_parameters(self):
        actual = self.ptr.parameters
        expected = {
            'threshdistr': 2,
            'outputunvoiced': 0,
            'precisetime': 0,
            'lowampsuppression': 0.1
        }
        self.assertEqual(expected, actual)

    def test_pyin(self):
        y = np.sin(2.0*np.pi*440.0*np.arange(0, 44100)/44100.0)
        fs = 44100
        times, pitch = self.ptr.pyin(y, fs)
        self.assertEqual(len(times), len(pitch))
        self.assertEqual(len(times), 168)
        self.assertEqual(times[0], 0.0)
        self.assertAlmostEqual(pitch[0], 62.857716, places=4)

    def test_run_from_file(self):
        times, pitch = self.ptr.run_from_file(AUDIO_FILE)
        self.assertEqual(len(times), len(pitch))
        self.assertEqual(len(times), 411)
        self.assertEqual(times[0], 0.0)
        self.assertAlmostEqual(pitch[0], 235.17456055)

    def test_run_from_audio(self):
        y = np.sin(2.0*np.pi*440.0*np.arange(0, 44100)/44100.0)
        fs = 44100
        times, pitch = self.ptr.run_from_audio(y, fs)
        self.assertEqual(len(times), len(pitch))
        self.assertEqual(len(times), 168)
        self.assertEqual(times[0], 0.0)
        self.assertAlmostEqual(pitch[0], 62.857716, places=4)
