# -*- coding: utf-8 -*-
""" Core methods and base class definitions
"""
import six


PITCH_TRACKER_REGISTRY = {}


class MetaPitchTracker(type):
    """Meta-class to register the available pitch trackers."""
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        # Register classes that inherit from the base class PitchTracker
        if "PitchTracker" in [base.__name__ for base in bases]:
            PITCH_TRACKER_REGISTRY[cls.get_id()] = cls
        return cls


class PitchTracker(six.with_metaclass(MetaPitchTracker)):
    """This class is an interface for all the pitch trackers available in
    massage. Each pitch tracker instance must inherit from it and implement the
    following method:
        - ``run_from_file``
            This takes an audio filepath and returns the estimated pitch `pitch` 
            and corresponding time stamps `times`
        - ``run_from_audio``
            This takes an audio signal in memory and returns the estimated pitch
            `pitch` and corresponding time stamps `times`
    """
    def __init__(self):
        pass

    def run_from_file(self, audio_filepath):
        """Run pitch tracker on an individual file."""
        raise NotImplementedError("This method must contain the implementation "
                                  "of the pitch tracker for a filepath")

    def run_from_audio(self, y, fs):
        """Run pitch tracker on an individual file."""
        raise NotImplementedError("This method must contain the implementation "
                                  "of the pitch tracker for a filepath")

    @classmethod
    def get_id(cls):
        """Method to get the id of the pitch tracker"""
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
    """This class is an interface for all the pitch trackers available in
    massage. Each pitch tracker instance must inherit from it and implement the
    following method:
        - ``run_from_file``
            This takes an audio filepath and saves the resynthesized output
        - ``run_from_audio``
            This takes an audio signal in memory and returns the estimated pitch
            `pitch` and corresponding time stamps `times`
    """
    def __init__(self):
        pass

    def run_from_file(self, audio_filepath, output_path,
                      times=None, pitch=None):
        """Run resynthesizer on an individual file."""
        raise NotImplementedError("This method must contain the implementation "
                                  "of the resynthesizer for a filepath")

    def run_from_audio(self, y, fs, y_out, times=None, pitch=None):
        """Run resynthesizer on an individual file."""
        raise NotImplementedError("This method must contain the implementation "
                                  "of the resynthesizer for a filepath")

    @classmethod
    def get_id(cls):
        """Method to get the id of the resynthesizer"""
        raise NotImplementedError("This method must return a string identifier"
                                  "of the pitch tracker")
