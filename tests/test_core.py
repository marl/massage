"""Tests for massage/core.py
"""
import unittest
import medleydb as mdb
import numpy as np

from massage import core


class TestTranscriptionRegistry(unittest.TestCase):

    def test_keys(self):
        actual = sorted(core.TRANSCRIBER_REGISTRY.keys())
        expected = sorted(['pyin'])
        self.assertEqual(expected, actual)

    def test_types(self):
        for val in core.TRANSCRIBER_REGISTRY.values():
            self.assertTrue(issubclass(val, core.Transcriber))


class TestTranscription(unittest.TestCase):

    def setUp(self):
        self.ptr = core.Transcriber()

    def test_run(self):
        y = np.zeros((1000, 1))
        fs = 44100
        with self.assertRaises(NotImplementedError):
            self.ptr.run(y, fs)

    def test_get_id(self):
        with self.assertRaises(NotImplementedError):
            self.ptr.get_id()


class TestResynthesizerRegistry(unittest.TestCase):

    def test_keys(self):
        actual = sorted(core.RESYNTHESIZER_REGISTRY.keys())
        expected = sorted([])
        self.assertEqual(expected, actual)

    def test_types(self):
        for val in core.RESYNTHESIZER_REGISTRY.values():
            self.assertTrue(issubclass(val, core.Resynthesizer))


class TestResynthesizer(unittest.TestCase):

    def setUp(self):
        self.rsyn = core.Resynthesizer()

    def test_run_from_file(self):
        audio_file = "data/vocal.wav"
        output_path = "data/vocal_resynth.wav"
        with self.assertRaises(NotImplementedError):
            self.rsyn.run_from_file(audio_file, output_path)

    def test_run_from_audio(self):
        y = np.zeros((1000, 1))
        fs = 44100
        with self.assertRaises(NotImplementedError):
            self.rsyn.run_from_audio(y, fs)

    def test_get_id(self):
        with self.assertRaises(NotImplementedError):
            self.rsyn.get_id()


class TestRemixerRegistry(unittest.TestCase):

    def test_keys(self):
        actual = sorted(core.REMIXER_REGISTRY.keys())
        expected = sorted([])
        self.assertEqual(expected, actual)

    def test_types(self):
        for val in core.REMIXER_REGISTRY.values():
            self.assertTrue(issubclass(val, core.Remixer))


class TestRemixer(unittest.TestCase):

    def setUp(self):
        self.rmx = core.Remixer()

    def test_remix(self):
        mtrack = mdb.MultiTrack("LizNelson_Rainfall")
        output_audio_path = "data/rainfall_remix.wav"
        output_jams_path = "data/rainfall_remix.jams"
        with self.assertRaises(NotImplementedError):
            self.rmx.remix(mtrack, output_audio_path, output_jams_path)

    def test_get_id(self):
        with self.assertRaises(NotImplementedError):
            self.rmx.get_id()
