import json
import os
from random import choice

import jams
import librosa
import numpy as np
import scipy
from massage.resynth import SF_PATH, ACOUSTIC_SF_MFCC


def compute_avg_mfcc(fpath=None, y=None, sr=None):
    """ Compute the average mfcc of a signal y

    Parameters
    ----------
    fpath: str, optional, default: None
        path to an audio file.
    y: ndarray, optional, default: None
        array containing the audio signal, has to be mono!
    sr: float
        sample rate

    Returns
    -------
    avg_mfcc: ndarray
        a vector containing the average mfccs over time. discarding the
        first bin.
    """
    if fpath is not None:
        y, sr = librosa.load(fpath, mono=True)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    avg_mfcc = np.mean(mfccs, axis=1)[1:]
    return avg_mfcc


def onset_offset(y=None, sr=None, hop_length=512,
                 feature=librosa.feature.melspectrogram):
    """ given the a signal, compute the onsets and offsets based on feature

    Parameters
    ----------
    y: ndarray, default: None
        mono signal
    sr: float, default: None
        sample rate
    hop_length: int, default: 512
        hop_length associated with feature computation
    feature: function, default: librosa.feature.melspectrogram
        feature function to be used in computing onsets.

    Returns
    -------
    onsets_t: ndarray
        array of times indicating the detected onsets
    offsets_t: ndarray
        array of times indicating the detected offsets
    onset_strength: ndarray
        strenth of the onsets
    """
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length)
    X = feature(y=y, sr=sr, hop_length=hop_length)
    boundaries = librosa.segment.subsegment(X, onsets, n_segments=2)

    # Throw out everything before the first onset
    boundaries = boundaries[boundaries > np.min(onsets)]
    offsets = np.setdiff1d(boundaries, onsets)
    onset_strength = librosa.onset.onset_strength(S=X)
    onsets_t = librosa.core.frames_to_time(
        onsets, hop_length=hop_length, sr=sr)
    offsets_t = librosa.core.frames_to_time(
        offsets, hop_length=hop_length, sr=sr)
    return onsets_t, offsets_t, onset_strength[onsets]


def compute_envelope(y_input, thresh=0.01, lpf_cutoff=0.03, alpha=20.0,
                     win_length=4096, theta=0.15):
    """ compute envelope of a single channel signal

    Parameters
    ----------
    y_input: ndarray
        mono signal
    thresh: float, default: 0.01
        threshold for low energy
    lpf_cutoff: float, default: 0.01
        smoothing filter on y, in radian
    alpha: float, default: 20.0
        Controls the steepness of the envelope via sigmoid
        higher alpha gives a steeper envelope transition
        lower alpha gives a more gradual envelop transition
    win_length: int, default: 4096
        window size for doing stft analysis
    theta: float, default: 0.15
        bias on the smoothed signal in the context of logistic function
        higher theta reduces envelope activation sensitivity
        lower theta increases envelope activation sensitivity

    Returns
    -------
    y_env: ndarray
        a vector specifying the amplitude envelope
    """
    S = librosa.stft(
        y_input, n_fft=win_length, hop_length=win_length,
        win_length=win_length)
    S_samples = librosa.core.frames_to_samples(
        range(len(S[0])), hop_length=win_length)
    y_smooth = np.mean(np.abs(S), axis=0)

    # normalization (to overall energy)
    if np.max(np.abs(y_smooth)) > 0:
        y_smooth = y_smooth / np.max(np.abs(y_smooth))

    # binary thresholding for low overall energy events
    y_smooth[y_smooth < thresh] = 0

    # LP filter
    b_coeff, a_coeff = scipy.signal.butter(2, lpf_cutoff, 'low')
    y_smooth = scipy.signal.filtfilt(b_coeff, a_coeff, y_smooth)

    # logistic function to semi-binarize the output; confidence value
    y_conf = 1.0 - (1.0 / (1.0 + np.exp(np.dot(alpha, (y_smooth - theta)))))

    energy_interpolator = scipy.interpolate.interp1d(
        S_samples, y_conf, bounds_error=False, fill_value='extrapolate')
    y_env = energy_interpolator(np.arange(len(y_input)))

    return y_env


def get_energy_envelope(y):
    """ Generate the energy envelope per channel of a signal y

    Parameters
    ----------
    y: ndarray
        the audio signal in questions. y.shape = num_ch * samples

    Returns
    -------
    energy_array: ndarray
        energy envelope of the audio signal
    """
    energy_list = []
    max_amplitudes = np.max(np.abs(y), 1)
    for ch in range(y.shape[0]):
        energy_ch = compute_envelope(y[ch, :])
        if np.max(energy_ch) > 0.0:
            energy_ch = max_amplitudes[ch] * energy_ch / np.max(energy_ch)
        energy_list.append(energy_ch)
    energy_array = np.array(energy_list)
    return energy_array


def pick_sf(y, sr, instrument):
    """ based on the audio signal, pick the closest soundfont based on
        average mfcc.

    Parameters
    ----------
    y: ndarray of float
        audio signal, y.shape = num_ch * samples
    sr: float
        sampling rate of the audio
    instrument : str
        label of the instrument family. This is supplied to pick which group
        of mfccs this algorithm will look in.

    Returns
    -------
    soundfont_path: str
        the path to the sf2 file
    program: int
        the MIDI program number to use for this specific soundfont
    """
    y_mono = librosa.core.to_mono(y)
    avg_mfcc = compute_avg_mfcc(y=y_mono, sr=sr)

    # add more cases as we develop?
    if 'acoustic guitar' in instrument:
        sf_mfcc = ACOUSTIC_SF_MFCC
    else:
        raise ValueError("invalid instrument {}".format(instrument))

    # SF Matching
    z_avg = np.mean(sf_mfcc['matrix'], axis=0)
    z_std = np.std(sf_mfcc['matrix'], axis=0)
    z_mat = (sf_mfcc['matrix'] - z_avg) / z_std
    z_current = (avg_mfcc - z_avg) / z_std

    mfcc_diff = np.linalg.norm(z_mat - z_current, axis=1)
    min_diff_idx = np.argmin(mfcc_diff)

    program = sf_mfcc['programs'][min_diff_idx]
    soundfont_name = sf_mfcc['soundfonts'][min_diff_idx]
    soundfont_path = os.path.join(SF_PATH, "{}.sf2".format(soundfont_name))
    return soundfont_path, program


def amplitude_to_velocity(energies):
    """ Map amplitude values to sensible velocities

    Parameters
    ----------
    energies : ndarray of float
        an array of energy values

    Returns
    -------
    velocities : ndarray of int
        an array of velocities in the range [60,120]
        The range is chosen to be restricted on purpose
    """
    velocities = 40.0 * (energies / np.max(energies)) + 80.0
    velocities = np.round(velocities).astype(int)
    if any(velocities > 120) or any(velocities < 60):
        velocities[velocities > 120] = 120
        velocities[velocities < 60] = 60
    return velocities


def midi_to_jams(midi_data):
    """ Turn PrettyMIDI objects into JAMS objects. Notes only.
        Velocities are mapped to confidence, [0,127] -> [0.0,1.0]
        Each instrument is mapped to a new annotation.

    Parameters
    ----------
    midi_data: PrettyMIDI
        MIDI data with notes.

    Returns
    -------
    jam: JAMS
        JAMS objects with notes saved as annotation
    """
    # Get all the note events from the first instrument
    jam = jams.JAMS()
    for inst_idx in range(len(midi_data.instruments)):
        solo_inst = midi_data.instruments[inst_idx]
        pm_notes = solo_inst.notes  # assuming midi file is single insturment
        inst_dur = solo_inst.get_end_time()
        if jam.file_metadata.duration < inst_dur:
            jam.file_metadata.duration = inst_dur
        # Create annotation container for the notes.
        jam_notes = jams.Annotation(
            namespace='pitch_midi', time=0, duration=inst_dur)
        for pm_note in pm_notes:
            jam_notes.append(
                time=pm_note.start, value=pm_note.pitch,
                duration=pm_note.end - pm_note.start,
                confidence=pm_note.velocity / 127.0)

        # Associate the annotation with the container
        jam.annotations.append(jam_notes)
    return jam


def voicing_dist(previous_voicing, voicing_candidate):
    """ Find the 'distance' between the previous voicing and the candidate.

    Parameters
    ----------
    previous_voicing : list of int
        list elements should be MIDI values from 0 to 127, indicating
        pitches in a chord voicing

    voicing_candidate : list of int
        list elements should be MIDI values from 0 to 127, indicating
        pitches in a chord voicing

    Returns
    -------
    dist : float
        average of min distance between notes
    """
    previous_voicing = np.array(previous_voicing)
    voicing_candidate = np.array(voicing_candidate)
    note_dists = np.zeros(len(previous_voicing))
    for i, note in enumerate(previous_voicing):
        can_dist = np.abs(note - voicing_candidate)
        value = voicing_candidate[np.where(can_dist == can_dist.min())]
        if len(value) > 1:
            value = value[0]
        note_dists[i] = np.abs(note - value)
    return np.mean(note_dists)


def get_all_voicings(voicing_file):
    """ Load chord voicings

    Parameters
    ----------

    voicing_file : str
        path to json file of voicings

    Returns
    -------
    voicing : dict
        keys are chord names, vals are lists of voicings (lists).
        Each voicing is a list of up to length 6 of midi note numbers.
        All voicings that include notes lower than E2 are discarded.
    """
    with open(voicing_file, 'r') as f_handle:
        voicings = json.load(f_handle)

    min_note = librosa.note_to_midi('E2')
    for k in voicings.keys():
        chords = []
        for voicing in voicings[k]:
            if np.min(voicing) >= min_note:
                chords.append(sorted(list(set(voicing))))
        voicings[k] = chords
    return voicings


def choose_voicing(chord_name, voicings, prev_voicing=None):
    """ Given a chord name, a set of possible voicings, and the previous
        voicing, choose the best voicing.

    Parameters
    ----------
    chord_name : str
        chord name of the form C:maj6, G:dim7, etc.
    voicings : dict
        dictionary of possible voicings
    prev_voicing : list of int
        Optional - previous voicing.

    Returns
    -------
    voicing : list of int
        best voicing for the given chord name.
        List members are int midi values
    """
    chord_parts = chord_name.split(':')
    if chord_parts[0] == 'D#':
        chord_parts[0] = 'Eb'
    elif chord_parts[0] == 'G#':
        chord_parts[0] = 'Ab'
    elif chord_parts[0] == 'A#':
        chord_parts[0] = 'Bb'

    chord_name = '{}:{}'.format(chord_parts[0], chord_parts[1])
    voicing_candidates = voicings[chord_name]
    if prev_voicing is not None:
        cand_dist = np.zeros(len(voicing_candidates))
        for i, cand in enumerate(voicing_candidates):
            cand_dist[i] = voicing_dist(prev_voicing, cand)
        voicing = voicing_candidates[np.argmin(cand_dist)]
    else:
        voicing = choice(voicing_candidates)
    return voicing
