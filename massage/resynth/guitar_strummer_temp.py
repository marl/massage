import jams
import os
import pretty_midi
import json
from random import choice
import numpy as np
import librosa
import csv
import medleydb as mdb
import scipy
import sox
import matplotlib.pyplot as plt
import tempfile

SF_PATH = '../resources/sf2'
VOICING_FILE = '../resources/chord_voicings.json'


def get_all_voicings(voicing_file):
    """ Load chord voicings
    Args:
        voicing_file (str): path to json file of voicings
    Returns:
        voicing (dict): keys are chord names, vals are lists of voicings.
            Each voicing is a list of up to length 6 of midi note numbers.
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


def voicing_dist(previous_voicing, voicing_candidate):
    """ Find the 'distance' between the previous voicing and the candidate.
    Args:
        previous_voicing (list): previous voicing
        voicing_candidate (list): current voicing candidate
    Returns:
        dist (float): average of min distance between notes
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


def choose_voicing(chord_name, voicings, prev_voicing=None):
    """ Given a chord name, a set of possible voicings, and the previous
    voicing, choose the best voicing.
    Args:
        chord_name (str): chord name of the form C:maj6, G:dim7, etc.
        voicings (dict): dictionary of possible voicings
        prev_voicing (list): Optional - previous voicing.
    Returns:
        voicing (list): best voicing for the given chord name
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


def get_strum(start_t, end_t, chord, voicings, prev_voicing,
              backwards=False, min_duration=0.1, velocity=None):
    strum = []

    notes = choose_voicing(chord, voicings, prev_voicing)
    strum_duration = end_t - start_t

    if strum_duration > 2.5:
        end_t = start_t + 2.5
        strum_duration = 2.5

    prev_voicing = notes
    shifted_start = start_t

    # USING FIXED VELOCITY RANGE HERE BECAUSE WE'RE ADAPTING
    # THE OVERALL ENVELOPE LATER!
    velocity = None
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
        raise ValueError("Something terrible happened! max_delay is {}".format(max_delay))

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

    return strum, prev_voicing


# NoteOn before NoteOff on the smae note problem...
def generate_chord_midi(chord_sequence, onsets, offsets, velocities,
                        program, energy_interp, voicing_file):
    """ Given list of triples of the form (start_time, end_time, chord_name),
        generate midi file.
    Args:
        chord_sequence (list): list of triples of the form
            (start_time, end_time, chord_name), with start_time, end_time in
            seconds, and chord names of the form 'A:min6', 'Bb:maj', etc
        instrument (str): General Mid00i instrument name. Defaults to piano.
    Returns:
        Nothing
    """
    voicings = get_all_voicings(voicing_file)
    midi_chords = pretty_midi.PrettyMIDI()

    chords = pretty_midi.Instrument(program=program)

    prev_voicing = None
    for triple in chord_sequence[:-1]:
        start_t = triple[0]
        end_t = triple[1]

        if triple[2] == 'N' or triple[2] == 'X':
            continue

        onsets_idx_in_range = np.where(
            np.logical_and(onsets >= start_t, onsets < end_t))[0]

        # shift onset back by 10 ms to adjust for strum start
        start_times = [onsets[i] - 0.01 for i in onsets_idx_in_range]

        energies = [np.max([energy_interp[0](s), energy_interp[1](s)]) for s in start_times]

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
                end_t = np.random.uniform(start_t + min_duration, next_start_t)
            else:
                end_t = np.random.uniform(end_t, next_start_t)

            strum, prev_voicing = get_strum(
                start_t, end_t, triple[2], voicings, prev_voicing,
                backwards=backwards, velocity=vel_values[i]
            )
            chords.notes.extend(strum)

    midi_chords.instruments.append(chords)

    return midi_chords


def compute_avg_mfcc(fpath=None, y=None, sr=None):
    if fpath is not None:
        y, sr = librosa.load(fpath)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    avg_mfcc = np.mean(mfccs, axis=1)[1:]
    return avg_mfcc


def midi_to_multif0(midi_data):
    fs = 22050.0 / 256.0
    piano_roll = midi_data.get_piano_roll(fs=fs)
    n_freqs, n_times = piano_roll.shape
    times = np.arange(n_times) / fs
    freqs_midi = np.arange(n_freqs)
    freqs_hz = librosa.midi_to_hz(freqs_midi)

    freqs_mf0 = [[] for t in times]
    for i, t in enumerate(times):
        f_idx = np.where(piano_roll[:, i] > 0)[0]
        for j in f_idx:
            freqs_mf0[i].append(freqs_hz[j])

    return times, freqs_mf0


def save_multif0(fpath, times, freqs_mf0):
    with open(fpath, 'w') as fhandle:
        csv_writer = csv.writer(fhandle, delimiter='\t')
        for t, f in zip(times, freqs_mf0):
            row = [t]
            row.extend(f)
            csv_writer.writerow(row)


def onset_offset(y=None, sr=None, hop_length=512,
                 feature=librosa.feature.melspectrogram, **kwargs):
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


def compute_envelope(y_input, thresh=0.01, lpf_cutoff=0.01, alpha=20.0,
                     win_length=4096, theta=0.15, compression=0.5):
    S = librosa.stft(y_input, n_fft=win_length, hop_length=win_length, win_length=win_length)
    S_samples = librosa.core.frames_to_samples(range(len(S[0])), hop_length=win_length)
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


def get_energy_envelope(amplitude, y, win_length=2048 * 8):
    for ch in range(y.shape()[0]):
        energy_ch = compute_envelope(y[ch, :])
        energy_list = []
        if np.max(energy_ch) > 0.0:
            energy_ch = amplitude[0] * energy_ch / np.max(energy_ch)
            energy_list.append(energy_ch)

    return energy_list


def pick_sf(y, sr, instrument):
    y_mono = librosa.core.to_mono(y)
    avg_mfcc = compute_avg_mfcc(y=y_mono, sr=sr)

    if 'acoustic guitar' in instrument:
        sf_mfcc = np.load("../resources/mfcc_npz/acoustic_sf_mfcc.npz")
    elif 'clean electric guitar' in instrument:
        sf_mfcc = np.load("../resources/mfcc_npz/clean_electric_sf_mfcc.npz")
    elif 'distorted electric guitar' in instrument:
        sf_mfcc = np.load("../resources/mfcc_npz/distorted_electric_sf_mfcc.npz")
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


def stem_loader(stem_path, instrument):
    print(stem_path)
    y, sr = librosa.load(stem_path, sr=44100, mono=False)
    num_ch = y.shape[0]
    print(num_ch)
    amplitudes = []
    for ch in range(num_ch):
        print(ch)
        tfm = sox.Transformer()
        tfm.remix(remix_dictionary = {1: [ch+1]})
        stats = tfm.stat(stem_path)
        amplitude = float(stats['Maximum amplitude'])
        amplitudes.append(amplitude)
    return y, sr, amplitudes


def amplitude_to_velocity(energies):
    velocities = 40.0 * (energies / np.max(energies)) + 80.0
    velocities = np.round(velocities).astype(int)
    if any(velocities > 127) or any(velocities < 0):
        velocities[velocities > 127] = 120
        velocities[velocities < 0] = 60
    return velocities


def write_small_wav(save_path, y, fs=44100, bitdepth=16):
    fhandle, tmp_file = tempfile.mkstemp(suffix='.wav')
    librosa.output.write_wav(tmp_file, y, fs)
    tfm = sox.Transformer()
    tfm.convert(bitdepth=bitdepth)
    tfm.build(tmp_file, save_path)
    os.close(fhandle)
    os.remove(tmp_file)


def midi_to_jams(midi_data):
    # Get all the note events from the first instrument
    solo_inst = midi_data.instruments[0]
    pm_notes = solo_inst.notes  # assuming midi file is single insturment

    jam = jams.JAMS()
    jam.file_metadata.duration = solo_inst.get_end_time()

    # Create annotation container for the notes.
    jam_notes = jams.Annotation(namespace='pitch_midi', time=0, duration=jam.file_metadata.duration)
    for pm_note in pm_notes:
        t = pm_note.start
        dur = pm_note.end - pm_note.start
        value = pm_note.pitch
        jam_notes.append(time=pm_note.start, value=pm_note.pitch,
                         duration=pm_note.end - pm_note.start)

    # Associate the annotation with the container
    jam.annotations.append(jam_notes)

    return jam


def resynth_guitar_rabitt(jams_file, audio_path=None, instrument='acoustic guitar', fs=44100):
    # jams file should contain chord annotation
    # Loading and Initilization
    jam = jams.load(jams_file)

    fileparts = os.path.basename(jams_file).split('_')
    track_id = '_'.join(fileparts[:2])
    mtrack = mdb.MultiTrack(track_id)
    stem_id_str = fileparts[3].split('.')[0]
    stem_id = int(stem_id_str)

    # Get Chord annotation
    time_intervals, chord_labs = jam.search(namespace='chord')[0].to_interval_values()

    chord_sequence = []
    for interval, lab in zip(time_intervals, chord_labs):
        chord_sequence.append([interval[0], interval[1], lab])

    # Pick SF, envelop, and other param for resynth
    # (amplitude, program, sound_font,
    #  onsets, offsets, velocities, envelope) = get_stem_info(
    #     audio_path, instrument)
    y, sr, amplitude = stem_loader(audio_path, instrument)
    envelope = get_energy_envelope(amplitude, y)

    y_mono = librosa.core.to_mono(y)

    onsets, offsets, energies = onset_offset(y=y_mono, sr=sr)
    velocities = amplitude_to_velocity(energies)
    sound_font, program = pick_sf(y_mono, sr, instrument)

    print("soundfont: {} program: {}".format(sound_font, program))

    env_times = librosa.samples_to_time(range(len(envelope[0])), sr=44100)
    energy_interp_L = scipy.interpolate.interp1d(
        env_times, envelope[0],
        bounds_error=False, fill_value='extrapolate'
    )

    energy_interp_R = scipy.interpolate.interp1d(
        env_times, envelope[1],
        bounds_error=False, fill_value='extrapolate'
    )
    energy_interp = [energy_interp_L, energy_interp_R]

    # Generating midi for synthesis and also jams file
    midi_data = generate_chord_midi(
        chord_sequence, onsets, offsets, velocities, program, energy_interp, VOICING_FILE)
    # Convert midi_data -> jams file
    jams_out = midi_to_jams(midi_data)

    # Resynth
    y = midi_data.fluidsynth(sf2_path=sound_font, fs=fs)
    if len(y) == 0:
        print("y is none")
        return

    # Normalize
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # Plotting the energy envelop
    y_len = len(y)

    env_len = len(envelope[0])
    if y_len > env_len:
        y_stereo = np.array(
            [envelope[0] * y[:env_len], envelope[1] * y[:env_len]])

        plt.subplot(2, 1, 1)
        plt.plot(envelope[0], 'r')
        plt.plot(y_stereo[0])
        plt.ylim([-1, 1])

        plt.subplot(2, 1, 2)
        plt.plot(envelope[1])
        plt.plot(y_stereo[1])
        plt.ylim([-1, 1])

    else:
        y_stereo = np.array(
            [envelope[0][:y_len] * y, envelope[1][:y_len] * y])

        plt.subplot(2, 1, 1)
        plt.plot(envelope[0][:y_len], 'r')
        plt.plot(y_stereo[0])
        plt.ylim([-1, 1])

        plt.subplot(2, 1, 2)
        plt.plot(envelope[1][:y_len])
        plt.plot(y_stereo[1])
        plt.ylim([-1, 1])

    plt.show()

    return y_stereo, jams_out
