"""Test for massage.pitch.pyin
"""
import unittest
import os
import numpy as np

from massage.transcriber import pyin


def relpath(f):
    return os.path.join(os.path.dirname(__file__), f)


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

    def test_run(self):
        y = np.sin(2.0*np.pi*440.0*np.arange(0, 44100)/44100.0)
        fs = 44100
        jam = self.ptr.run(y, fs)
        ann = jam.annotations['pitch_hz', 0]
        print(ann.data[1].time)
        self.assertAlmostEqual(ann.data[1].value, 62.858, places=2)
