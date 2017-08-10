"""Guitar Resynthesiser using chord annotation and synthesize strums
"""

import librosa
import numpy as np
import pretty_midi
import scipy

import massage.resynth.util as util
from massage.core import Resynthesizer
from massage.resynth import VOICING_FILE


class GuitarStrummer(Resynthesizer):
    """ Resynthesize a stereo audio file that simulate a strumming guitar.
        Takes a guitar stem and associated JAMS object with chord annotation
        and generate strummed chords according to JAMS chord annotation and
        onsets detected from Audio File.
    """

    def run(self, y, fs, jam=None, instrument_label=None):
        """ run the resynthesizer

        Parameters
        ----------
        y : np.array
            an audio signal.
            shape[0] = number of channels
            shape[1] = number of samples
        fs : int
            the sample rate of the audio signal y
        jam : JAMS
            a jams object that contain 'chord' annotation for y
        instrument_label : str
            Label specifying the instrument of the track. Will add more
                -'acoustic guitar' : Acoustic Guitar
        """

        # Get Chord annotation
        time_intervals, chord_labs = jam.search(
            namespace='chord')[0].to_interval_values()

        chord_sequence = []
        for interval, lab in zip(time_intervals, chord_labs):
            chord_sequence.append([interval[0], interval[1], lab])

        envelope = util.get_energy_envelope(y)
        y_mono = librosa.core.to_mono(y)

        onsets, offsets, energies = util.onset_offset(y=y_mono, sr=fs)
        velocities = util.amplitude_to_velocity(energies)
        sound_font, program = util.pick_sf(y_mono, fs, instrument_label)

        # print("soundfont: {} program: {}".format(sound_font, program))

        env_times = librosa.samples_to_time(np.arange(len(envelope[0])), sr=fs)

        energy_interp = []
        # list of interp1d objects, len(.) = number of channels
        for ch in range(envelope.shape[0]):
            energy_interp_ch = scipy.interpolate.interp1d(
                env_times, envelope[0], bounds_error=False,
                fill_value='extrapolate')
            energy_interp.append(energy_interp_ch)

        # Generating midi for synthesis and also jams file
        midi_data = self._generate_chord_midi(
            chord_sequence, onsets, offsets, velocities, program,
            energy_interp, VOICING_FILE)
        # Convert midi_data -> jams file
        jams_out = util.midi_to_jams(midi_data) # audio duration

        # Resynth
        y = midi_data.fluidsynth(sf2_path=sound_font, fs=fs)
        if len(y) == 0:
            print("y is none")
            return

        # Normalize
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        y_len = len(y)
        env_len = len(envelope[0])
        if y_len > env_len:
            y_stereo = np.array(
                [envelope[0] * y[:env_len], envelope[1] * y[:env_len]])
        else:
            y_stereo = np.array(
                [envelope[0][:y_len] * y, envelope[1][:y_len] * y])

        return y_stereo, jams_out

    @classmethod
    def get_id(cls):
        """Method to get the id of the pitch tracker

        Returns
        -------
        str : 'guitar_strummer'
        """
        return 'guitar_strummer'

    def _get_strum(self, start_t, end_t, chord, voicings, prev_voicing,
                   backwards=False, velocity=None):
        """
        Parameters
        ----------
        start_t: float
            starting time of the strum
        end_t: float
            ending time of the strum
        chord: str
            'Ab:maj6', 'D#:min', etc
        voicings: dict
            {chord_string: list of voicing}
            voicing is a list of int
        prev_voicing: list of int
            a list of midi numbers to represent the previous voicing
        backwards: bool, default = False
            - False : strumming down
            - True  : strumming up
        velocity : int
            midi velocity value of the strum to be created

        Returns
        -------
        strum : list of pretty_midi.Note
            a list of pretty_midi.Note object with small delays to simulate
            strumming
        current_voicing : list of int
            a list of the pitches in midi number to specify voicing
        """
        strum = []

        notes = util.choose_voicing(chord, voicings, prev_voicing)
        strum_duration = end_t - start_t

        if strum_duration > 2.5:
            end_t = start_t + 2.5
            strum_duration = 2.5

        current_voicing = notes
        shifted_start = start_t

        # USING FIXED VELOCITY RANGE HERE BECAUSE WE'RE ADAPTING
        # THE OVERALL ENVELOPE LATER!
        if velocity is None:
            velocity = np.random.choice(range(70, 100))
        elif velocity < 70:
            velocity = 70
        elif velocity > 100:
            velocity = 100

        max_delay = strum_duration / float(len(notes))
        if max_delay > 0.05:
            all_choices = [0.02, 0.03, 0.04, 0.05]
        elif max_delay > 0.04:
            all_choices = [0.02, 0.03, 0.04]
        elif max_delay > 0.03:
            all_choices = [0.01, 0.02, 0.03]
        elif max_delay > 0.02:
            all_choices = [0.01, 0.02]
        elif max_delay >= 0.01:
            all_choices = [0.01]
        else:
            raise ValueError(
                "Something terrible happened! max_delay is {}".format(max_delay))

        if backwards:
            notes = notes[::-1]
            choices = all_choices[:np.random.choice(range(1, 3))]
        else:
            choices = all_choices[:np.random.choice(range(1, 5))]

        for note in notes:
            note = pretty_midi.Note(
                velocity=velocity + np.random.choice(range(-5, 5)),
                pitch=note, start=shifted_start, end=end_t
            )
            strum.append(note)
            shifted_start = shifted_start + np.random.choice(choices)

        return strum, current_voicing

    def _generate_chord_midi(self, chord_sequence, onsets, offsets, velocities,
                             program, energy_interp, voicing_file):
        """ Given list of triples of the form
            (start_time, end_time, chord_name), generate midi file.

        Parameters
        ----------
        chord_sequence: list
            list of triples of the form (start_time, end_time, chord_name),
            with start_time, end_time in seconds, and chord names of the
            form 'A:min6', 'Bb:maj', etc
        onsets: ndarray of float
            a list of onset times
        offsets: ndarray of float
            a lost of offset times
        velocities: ndarray of int
            array of midi velocity values for each chord
        program: int
            midi value for program number
        energy_interp: list of scipy.interpolate.interp1d
            a list of interpolator object (1 per channel) to facilitate energy
            interpolation
        voicing_file: str
            file path to the voicing.json file

        Returns
        midi_chords: pretty_midi.PrettyMIDI
            pretty_midi object ready to be synthesized

        """
        voicings = util.get_all_voicings(voicing_file)
        midi_chords = pretty_midi.PrettyMIDI()

        chords = pretty_midi.Instrument(program=program)

        prev_voicing = None
        for triple in chord_sequence:
            start_t = triple[0]
            end_t = triple[1]

            if triple[2] == 'N' or triple[2] == 'X':
                continue

            onsets_idx_in_range = np.where(
                np.logical_and(onsets >= start_t, onsets < end_t))[0]

            # shift onset back by 10 ms to adjust for strum start
            start_times = [onsets[i] - 0.01 for i in onsets_idx_in_range]

            energies = [np.max([energy_interp[0](s), energy_interp[1](s)])
                        for s in start_times]

            end_times = [offsets[i] for i in onsets_idx_in_range]
            if len(start_times) == 0:
                continue

            vel_values = [velocities[i] for i in onsets_idx_in_range]

            next_start_times = start_times[1:] + [end_t]
            zipper = zip(start_times, end_times, next_start_times, energies)

            for i, (start_t, end_t, next_start_t, energy) in enumerate(zipper):
                if energy < 0.03:
                    continue

                if i % 2 == 0:
                    backwards = False
                else:
                    backwards = True

                min_duration = 0.06
                if next_start_t - start_t < min_duration:
                    continue
                elif end_t - start_t < min_duration:
                    end_t = np.random.uniform(
                        start_t + min_duration, next_start_t)
                else:
                    end_t = np.random.uniform(end_t, next_start_t)

                strum, prev_voicing = self._get_strum(
                    start_t, end_t, triple[2], voicings, prev_voicing,
                    backwards=backwards, velocity=vel_values[i]
                )
                chords.notes.extend(strum)

        midi_chords.instruments.append(chords)

        return midi_chords
