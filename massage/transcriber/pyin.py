"""Pyin pitch tracker
"""
import glob
import jams
import librosa
import numpy as np
import os
import vamp
from vamp import vampyhost

from massage.core import Transcriber


class Pyin(Transcriber):
    """probabalistic yin pitch tracker.

    Parameters
    ----------
    threshdistr : int, default=2
        Yin threshold distribution identifier.
            - 0 : uniform
            - 1 : Beta (mean 0.10)
            - 2 : Beta (mean 0.15)
            - 3 : Beta (mean 0.20)
            - 4 : Beta (mean 0.30)
            - 5 : Single Value (0.10)
            - 6 : Single Value (0.15)
            - 7 : Single Value (0.20)
    outputunvoiced : int, default=0
        Output estimates classified as unvoiced?
            - 0 : No
            - 1 : Yes
            - 2 : Yes, as negative frequencies
    precisetime : int, default=0
        If 1, use non-standard precise YIN timing (slow)
    lowampsuppression : float, default=0.1
        Threshold between 0 and 1 to supress low amplitude pitch estimates.

    """
    def __init__(self, threshdistr=2, outputunvoiced=0, precisetime=0,
                 lowampsuppression=0.1):
        """init method
        """
        Transcriber.__init__(self)

        self.parameters = {
            'threshdistr': threshdistr,
            'outputunvoiced': outputunvoiced,
            'precisetime': precisetime,
            'lowampsuppression': lowampsuppression
        }

    def run(self, y, fs):
        """Run pyin on an audio signal y.

        Parameters
        ----------
        y : np.array
            audio signal
        fs : float
            audio sample rate

        Returns
        -------
        jam : JAMS
            JAMS object with pyin output
        """
        output = vamp.collect(
            y, fs, 'pyin:pyin', output='smoothedpitchtrack',
            parameters=self.parameters
        )
        hop = float(output['vector'][0])
        freqs = np.array(output['vector'][1])
        times = np.arange(0, hop*len(freqs), hop)
        confidences = np.ones(())
        jam = jams.JAMS()
        jam.file_metadata.duration = len(y) / float(fs)
        ann = jams.Annotation(namespace='pitch_hz', time=0, duration=jam.file_metadata.duration)
        for time, freq in zip(times, freqs):
            ann.append(time=time, value=freq, duration=0, confidence=None)

        jam.annotations.append(ann)
        return jam

    @property
    def tasks(self):
        return ['pitch']

    @classmethod
    def get_id(cls):
        """Method to get the id of the pitch tracker"""
        return 'pyin'
