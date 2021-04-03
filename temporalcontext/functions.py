import os
import numpy as np
import tensorflow as tf
import json
import csv
import soundfile as sf
from scipy.signal import spectrogram
from scipy.signal.windows import hann


def read_selmap(selmap_file):
    """
    Read 3-entry lines from csv file describing the dataset.
    """

    with open(selmap_file, 'r', newline='') as f:
        return [entry for entry in csv.reader(f) if len(entry) == 3]


def read_folds_info(folds_info_file):
    """
    Read lines containing 2 sets of comma-separated integers, with sets separated by '|'.
    """

    with open(folds_info_file, 'r') as f:
        lines = [l[:-1] for l in f.readlines() if '|' in l]

    return [dict(
        train=[int(n) for n in tr_str.split(',')],
        test=[int(n) for n in ts_str.split(',')]
    ) for (tr_str, ts_str) in (l.split('|') for l in lines)]


def load_annotations(annot_filepath):
    """
    Fetch annotations (sorted based on start time) from a 'Raven selection table' file.
    Returns an Nx2 numpy array of start-end time pairs.
    """

    with open(annot_filepath, 'r', newline='') as seltab_file_h:
        csv_reader = csv.reader(seltab_file_h, delimiter='\t')

        col_headers = next(csv_reader)

        s_idx = col_headers.index('Begin Time (s)')
        e_idx = col_headers.index('End Time (s)')

        annots = np.asarray([
            [float(entry[s_idx]), float(entry[e_idx])]
            for entry in csv_reader])

    # Sort based on Begin Time
    return annots[np.argsort(annots[:, 0]), :]


def prepare_cnn_inputs(au_filepath, pos_annot_filepath, neg_annot_filepath,
                       seg_len_s, seg_advance_s,
                       pos_segments_path, neg_segments_path,
                       specgram_params, bandwidth_extents):

    # Load audio and break up into segments
    segments, seg_offsets, fs = load_audio_as_segments(au_filepath, seg_len_s, seg_advance_s)
    seg_times = np.concatenate([[seg_offsets], [seg_offsets + seg_len_s]], axis=0).T  # start & end pairs

    # Load annotations
    pos_annots = load_annotations(pos_annot_filepath)
    neg_annots = load_annotations(neg_annot_filepath)

    retval = []

    for selections, out_filepath in zip([pos_annots, neg_annots],
                                        [pos_segments_path, neg_segments_path]):

        annot_dur = selections[:, 1] - selections[:, 0]

        # The minimum of annotation durations or clip length, for each clip
        whole_durations = np.minimum(annot_dur, seg_len_s)

        # Mask of overlapping clips having enough overlap amount. A clip must have the minimum required overlap
        # with at least one annotation.
        overlaps_mask = np.stack(
            [(np.minimum(selections[:, 1], seg_times[idx, 1]) -
              np.maximum(selections[:, 0], seg_times[idx, 0])) >= whole_durations
             for idx in range(len(seg_offsets))])

        # Clips that were matched with one or more annotations
        valid_clips_mask = np.any(overlaps_mask, axis=1)  # at least one True in each row

        if np.any(valid_clips_mask):
            specgrams = segments2specgrams(segments[valid_clips_mask, ...], fs, specgram_params, bandwidth_extents)

            rel_dir, _ = os.path.split(out_filepath)
            os.makedirs(os.path.join(rel_dir), exist_ok=True)
            np.savez_compressed(out_filepath, segments=specgrams)

        retval.append(valid_clips_mask.sum())

    return retval   # Return counts per class


def load_audio_as_segments(au_filepath, seg_len_s, seg_advance_s, extents=None):

    # Load audio from file
    if extents is None:
        data, fs = sf.read(au_filepath, dtype='float32')
    else:
        fs = sf.info(au_filepath).samplerate
        # print([int(extents[0] * fs), int(extents[1] * fs) + 1])
        data, _ = sf.read(au_filepath, dtype='float32',
                          start=int(extents[0] * fs), stop=int(extents[1] * fs) + 1)

    # Convert params from sec to num samples
    seg_len = int(round(seg_len_s * fs))
    seg_advance = int(round(seg_advance_s * fs))

    seg_overlap = seg_len - seg_advance

    # Split up into chunks by creating a strided array (http://stackoverflow.com/a/5568169).
    # Technique copied from numpy's spectrogram() implementation.
    shape = data.shape[:-1] + ((data.shape[-1] - seg_overlap) // seg_advance, seg_len)
    strides = data.strides[:-1] + (seg_advance * data.strides[-1], data.strides[-1])
    sliced_data = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides, writeable=False)
    seg_start_samples = np.arange(0, len(data) - seg_len + 1, seg_advance, dtype=np.int)

    # Remove DC
    sliced_data = sliced_data - sliced_data.mean(axis=1, keepdims=True)
    # Bring to range [-1.0, 1.0]
    sliced_data = sliced_data / np.maximum(np.abs(sliced_data).max(axis=1, keepdims=True), 1e-24)

    return sliced_data, seg_start_samples / fs, fs


def segments2specgrams(segments, fs, specgram_params, bandwidth_extents):

    f, _, tf_rep = spectrogram(segments, fs=fs,
                               window=hann(specgram_params['nfft']), nperseg=specgram_params['nfft'],
                               noverlap=specgram_params['noverlap'], nfft=specgram_params['nfft'],
                               detrend=False, mode='psd')

    # Find out the indices of where to clip the bandwidth
    valid_f_idx_start = f.searchsorted(bandwidth_extents[0], side='left')
    valid_f_idx_end = f.searchsorted(bandwidth_extents[1], side='right') - 1

    # Clip bandwidth and return dB scale spectrograms
    return (10 * np.log10(tf_rep[:, valid_f_idx_start:(valid_f_idx_end + 1), :] + 1e-10)).astype(np.float32)


def get_testing_cnn_model(model_path):

    # Load base model
    base_model = tf.keras.models.load_model(model_path)

    # Scale scores for the positive class outputs.
    frac = 1.0 / float(base_model.output.shape[1])
    scaled_score = tf.maximum(
        0.0,
        (tf.slice(base_model.output, [0, 1], [-1, 1]) - frac) / np.float32(1.0 - frac))
    # The indexing in slice() assumes that there were only 2 classes and the 2nd is the positive class.

    # Create output subnetwork (from base) to output both 'score' and 'embedding'
    return tf.keras.Model(
        base_model.input,
        [scaled_score, base_model.get_layer('FC-D1').output])


def get_lstm_model_filename(mtype, time_steps, pp):
    # Fields include lstm type (t1/t2/t3), time steps, prediction point
    return 'lstm_{:s}_TS{:d}_PP{:d}_classifier.h5'.format(mtype, time_steps, pp)


def song_section_selections(selections, file_dur, max_call_separation, min_calls_in_song, seg_padding):

    sep_boundaries = np.where((selections[1:, 0] - selections[:-1, 0]) >= max_call_separation)[0]

    song_start_end_sel_idxs = \
        np.asarray([[0, len(selections) - 1]]) if len(sep_boundaries) == 0 else \
        np.stack(
            [
                np.concatenate([[0], sep_boundaries + 1]),
                np.concatenate([sep_boundaries, [len(selections) - 1]])
            ], axis=1)

    # Keep only songs with minimum required units in them
    song_start_end_sel_idxs = \
        song_start_end_sel_idxs[
            (song_start_end_sel_idxs[:, 1] - song_start_end_sel_idxs[:, 0] + 1) >= min_calls_in_song, :]

    # Determine section boundaries after adding some padding to song extents
    section_start_ends = np.stack(
        [
            np.maximum(0.0, selections[song_start_end_sel_idxs[:, 0], 0] - seg_padding),
            np.minimum(file_dur, selections[song_start_end_sel_idxs[:, 1], 1] + seg_padding)
        ], axis=1)

    return section_start_ends, song_start_end_sel_idxs


def non_song_section_selections(selections, min_non_song_duration):

    # Keep only selections lasting minimum required duration and get their extents
    section_start_ends = selections[(selections[:, 1] - selections[:, 0]) >= min_non_song_duration, :]

    return section_start_ends


def chunk_up(in_x, in_y, chunk_size, history_idx=0):
    """

    :param in_x: NxM array of N M-dimensional vectors
    :param in_y: N-length array
    :param chunk_size: Chunk up in_x into overlapping segments (with advance of 1) of length chunk_size each.
    :param history_idx: Index (non-negative) in in_y backwards from the end. When zero, the expected "y" for each chunk
        will be the in_y element corresponding to the end of the chunk. When non-zero, the expected "y" will be as many
        elements backward from the end of the chunk.
    :return: A 2-tuple containing
        C x chunk_size x M array of C chunks from in_x, and
        C-length array from in_y
    """

    assert chunk_size > 0 and 0 <= history_idx < chunk_size
    assert in_y is None or in_x.shape[0] == in_y.shape[0]

    return \
        np.stack([in_x[s_idx:(s_idx + chunk_size), ...]
                  for s_idx in range(0, in_x.shape[0] - chunk_size + 1)]), \
        None if in_y is None else in_y[(chunk_size - history_idx - 1):(in_y.shape[0] - history_idx)]


class LSTMData:

    @staticmethod
    def write(filepath, seg_start, seg_end, cnn_scores, cnn_fcn, y, selections=None):
        np.savez_compressed(
            filepath,
            seg_start_end=np.asarray([seg_start, seg_end], dtype=np.float32),
            cnn_scores=cnn_scores.astype(np.float32),
            cnn_fcn=cnn_fcn.astype(np.float32),
            y=y.astype(np.int8),
            selections=(selections[:, 0] if selections is not None else np.zeros((0, ))).astype(np.float32)
        )

    @staticmethod
    def read(filepath, ts=None, pp=None):
        """
        'ts' = time steps
        'pp' = prediction point
        If 'ts' and 'pp' are not None, only those values will be returned that would be common for all possible values
        of 'pp'.
        """

        with np.load(filepath) as d:
            seg_start_end = d['seg_start_end']
            cnn_scores = d['cnn_scores']
            cnn_fcn = d['cnn_fcn']
            y = d['y']
            selections = d['selections']

        if ts is not None and pp is not None:
            # Keep only values that will be common for all possible values of pp
            k_start_idx = pp
            k_end_idx = cnn_scores.shape[0] - ts + pp + 1
            cnn_scores = cnn_scores[k_start_idx:k_end_idx, ...]
            cnn_fcn = cnn_fcn[k_start_idx:k_end_idx, ...]
            y = y[k_start_idx:k_end_idx]

        return seg_start_end[0], seg_start_end[1], \
            cnn_scores, cnn_fcn, y.astype(np.float32), \
            selections


def gen_list_of_section_files(lstm_data_root, audio_files, section_suffixes):

    def is_required_file_type(filename, src_file):
        return (
            filename.endswith('.npz') and
            any([filename.startswith(src_file + suf) for suf in section_suffixes])
        )

    for audio_file in audio_files:
        audio_reldir, audio_basename = os.path.split(audio_file)

        annot_basedir = os.path.join(lstm_data_root, audio_reldir)
        audio_file = os.path.splitext(audio_basename)[0]

        for sec_file in [f for f in os.listdir(annot_basedir)
                         if is_required_file_type(f, audio_file)]:

            yield os.path.join(annot_basedir, sec_file)


def get_inputs_for_lstm(lstm_type, lstm_data_root, audio_files, ts, pp, section_suffixes, secondary_annots_info=None):

    assert lstm_type in ['t1', 't2', 't3']

    (secondary_annots_root, annot_duration, segment_length, seg_advance) = \
        secondary_annots_info if secondary_annots_info is not None \
        else (None, None, None, None)

    song_data_x = []
    song_data_y = []
    nonsong_data_x = []
    nonsong_data_y = []

    for sec_file in gen_list_of_section_files(lstm_data_root, audio_files, section_suffixes):

        sec_start, sec_end, t1_val, t2_val, y, _ = LSTMData.read(sec_file, ts, pp)

        if lstm_type == 't1':
            x = t1_val
        elif lstm_type == 't2':
            x = t2_val
        else:
            x = np.concatenate([t1_val, t2_val], axis=1)

        if x.shape[0] < ts:
            continue

        X, Y = chunk_up(x, y, ts, pp)

        if secondary_annots_info is not None:
            # Load secondary annotations
            added_annots_fullpath = os.path.join(secondary_annots_root, sec_file[len(lstm_data_root) + 1:])
            if os.path.exists(added_annots_fullpath):

                with np.load(added_annots_fullpath) as d:
                    added_annots = d['begin_time']
                # Only 'starts' were loaded. Add 'ends' and make an Nx2 array.
                added_annots = np.stack([added_annots, added_annots + annot_duration], axis=1)

                # Segment start & end times for each Y
                _, seg_start_ends = chunk_up(x, np.arange(x.shape[0]), ts, pp)
                seg_start_ends = sec_start + (pp * seg_advance) + (seg_start_ends * seg_advance)
                seg_start_ends = np.stack([seg_start_ends, seg_start_ends + segment_length], axis=1)

                # Keep only those added annots that occur within "valid" Ys' temporal extremes
                added_annots = added_annots[np.logical_and(added_annots[:, 0] >= seg_start_ends[0, 0],
                                                           added_annots[:, 1] <= seg_start_ends[-1, 1]), ...]

                match_mask = match_segments_to_selections(seg_start_ends, added_annots)

                # Update Y
                Y[match_mask] = 1

        if section_suffixes[1] in sec_file:
            nonsong_data_x.append(X)
            nonsong_data_y.append(Y)
        else:
            song_data_x.append(X)
            song_data_y.append(Y)

    return \
        np.concatenate(song_data_x, axis=0), np.concatenate(song_data_y, axis=0), \
        np.concatenate(nonsong_data_x, axis=0), np.concatenate(nonsong_data_y, axis=0)


def cnn_predict(cnn_model, segments, batch_size=1024):

    results = [
        cnn_model.predict(segments[s_idx:min(s_idx + batch_size, segments.shape[0]), ...])
        for s_idx in np.arange(0, segments.shape[0], batch_size)]

    # Concatenate 'scores' & 'embeddings' (independently) and return
    return np.concatenate([res[0] for res in results], axis=0), \
           np.concatenate([res[1] for res in results], axis=0)


def lstm_predict(lstm_model, x, ts, pp, batch_size=1024):

    X, _ = chunk_up(x, None, ts, pp)

    return np.concatenate(
        [lstm_model.predict(
            X[s_idx:min(s_idx + batch_size, X.shape[0]), ...])[:, 0]
         for s_idx in np.arange(0, X.shape[0], batch_size)], axis=0)


def match_segments_to_selections(seg_start_ends, selections, overlap_fraction_thld=1.0):
    """
    Returns a boolean mask (per segment) indicating whether or not a segment 'sufficiently' covers one or more
    selections. 'sufficiency' = at least overlap_fraction_thld of a selection's duration.
    """

    if selections.shape[0] > 0:
        overlap_thlds = (selections[:, 1] - selections[:, 0]) * overlap_fraction_thld

        return np.asarray([
            np.any((np.minimum(seg_e, selections[:, 1]) - np.maximum(seg_s, selections[:, 0])) >= overlap_thlds)
            for seg_s, seg_e in seg_start_ends], dtype=np.bool)
    else:
        # No selections; every segment should be of negative class
        return np.full((seg_start_ends.shape[0], ), False, dtype=np.bool)
