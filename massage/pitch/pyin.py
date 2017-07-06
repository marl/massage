"""Pyin pitch tracker
"""
import librosa
import numpy as np
import vamp

from massage.core import PitchTracker


class Pyin(PitchTracker):
    """probabalistic yin pitch tracker.
    """
    def __init__(self, threshdistr=2, outputunvoiced=0, precisetime=0,
                 lowampsuppression=0.1):
        """
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
        PitchTracker.__init__(self)

        self.parameters = {
            'threshdistr': threshdistr,
            'outputunvoiced': outputunvoiced,
            'precisetime': precisetime,
            'lowampsuppression': lowampsuppression
        }

    def pyin(self, y, fs):
        """Base call to pyin.
        Parameters
        ----------
        y : np.array
            audio signal
        fs : float
            audio sample rate
        """
        output = vamp.collect(
            y, fs, 'pyin:pyin', output='smoothedpitchtrack',
            parameters=self.parameters
        )
        hop = float(output['vector'][0])
        pitch = np.array(output['vector'][1])
        times = np.arange(0, hop*len(pitch), hop)
        return times, pitch

    def run_from_file(self, audio_filepath):
        """Run pitch tracker on an individual file."""
        y, fs = librosa.load(audio_filepath, sr=None)
        return self.pyin(y, fs)

    def run_from_audio(self, y, fs):
        """Run pitch tracker on an individual file."""
        return self.pyin(y, fs)

    @classmethod
    def get_id(cls):
        """Method to get the id of the pitch tracker"""
        return 'pyin'
