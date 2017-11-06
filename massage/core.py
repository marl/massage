# -*- coding: utf-8 -*-
""" Core methods and base class definitions
"""
import six


TRANSCRIBER_REGISTRY = {}
TRANSCRIBER_TASKS = ['pitch', 'notes', 'chords', 'onsets']

class MetaTranscriber(type):
    """Meta-class to register the available pitch trackers."""
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        # Register classes that inherit from the base class Transcriber
        if "Transcriber" in [base.__name__ for base in bases]:
            TRANSCRIBER_REGISTRY[cls.get_id()] = cls
        return cls


class Transcriber(six.with_metaclass(MetaTranscriber)):
    """This class is an interface for transcription-like methods available in
    massage. Transcribers can be, for example, pitch trackers, multif0
    detection algorithms, or chord estimation algorithms.

    Each transcriber instance must inherit from it and implement the
    following method:
        - ``run``
            This takes an audio signal y and returns a jams file containing
            the estimated transcription.
    """
    def __init__(self):
        pass

    def run(self, y, fs):
        """Run transcriber on an audio signal.

        Paramters
        ---------
        y : np.ndarray
            Audio signal
        fs : float
            Audio sample rate in Hz

        Returns
        -------
        jam : JAMS
            A jams file containing the output of the transcription.

        """
        raise NotImplementedError("This method must contain the implementation "
                                  "of the pitch tracker.")

    @property
    def tasks(self):
        """Property listing which trascription tasks are computed by this method.
        All elements of the list must be elements of TRANSCRIBER_TASKS

        Returns
        -------
        tasks : list
            List of tasks contained in jams file.
        """
        raise NotImplementedError("This method must return a list of the tasks "
                                  "that are output by this method.")

    @classmethod
    def get_id(cls):
        """Method to get the id of the transcriber"""
        raise NotImplementedError("This method must return a string identifier"
                                  "of the pitch tracker")


RESYNTHESIZER_REGISTRY = {}


class MetaResynthesizer(type):
    """Meta-class to register the available resynthesizers."""
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        # Register classes that inherit from the base class Resynthesizer
        if "Resynthesizer" in [base.__name__ for base in bases]:
            RESYNTHESIZER_REGISTRY[cls.get_id()] = cls
        return cls


class Resynthesizer(six.with_metaclass(MetaResynthesizer)):
    """This class is an interface for all the resynthesizers available in
    massage. Each resynthesizer instance must inherit from it and implement the
    following method:
        - ``run``
            This takes an audio signal, an optional JAMS object, and an optional
            instrument label and returns a resynthesized audio signal and
            corresponding JAMS object.
    """
    def __init__(self):
        pass

    def run(self, y, fs, jam=None, instrument_label=None):
        """Run resynthesizer on an audio singal.

        Parameters
        ----------
        y: np.ndarray
            Audio signal
        fs : float
            Audio samplerate
        jam : JAMS or None, default=None
            Jams file
        instrument_label : str or None, default=None
            Instrument label

        Returns
        -------
        y_resynth : np.ndarray
            Resynthesized audio signal sampled at input fs
        jam_resynth : JAMS
            Jams file with corresponding annotation(s)
        """
        raise NotImplementedError("This method must contain the implementation "
                                  "of the resynthesizer")

    @classmethod
    def get_id(cls):
        """Method to get the id of the resynthesizer"""
        raise NotImplementedError("This method must return a string identifier"
                                  "of the resynthesizer")


REMIXER_REGISTRY = {}


class MetaRemixer(type):
    """Meta-class to register the available remixers."""
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        # Register classes that inherit from the base class Remixer
        if "Remixer" in [base.__name__ for base in bases]:
            REMIXER_REGISTRY[cls.get_id()] = cls
        return cls


class Remixer(six.with_metaclass(MetaRemixer)):
    """This class is an interface for all the remixers available in
    massage. Each remixer instance must inherit from it and implement the
    following method:
        - ``remix``
            Takes a MultiTrack object as input and generates an audio and
            jams file.
    """

    def __init__(self):
        pass

    def remix(self, mtrack, output_audio_path, output_jams_path):
        """Remix a multitrack and return an audio and jams file
        """
        raise NotImplementedError("this method must contain the implementation "
                                  "of a remix method for a given multitrack.")

    @classmethod
    def get_id(cls):
        """Method to get the id of the remixer"""
        raise NotImplementedError("This method must return a string identifier"
                                  "of the remixer")
