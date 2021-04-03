
# Fullpath to raw data
raw_data_root = '/path/to/downloaded/extracted/audio_n_annots'

# Subdirectories under raw_data_root
raw_audio_dir = 'audio'
raw_annot_dir = 'annotations/raven_tables'
added_annot_dir = 'annotations/additional'


# Root directory path for the project
project_root = '/path/to/output/directory'

# Subdirectories under project_root (will be created as needed)
segments_dir = 'segments'
lstm_data_dir = 'lstm_data'
models_dir = 'models'
scores_dir = 'scores'
folds_dir = 'folds'
perf_results_dir = 'perf_results'


# ============================================================================


random_seed = 1337


# ----- Settings for processing data -----------------------------------------
#
segment_length = 4.0
annot_duration = 2.6
# For fs = 500 Hz, the below settings translate to 0.8 s window, 80% overlap
specgram_params = dict(nfft=400, noverlap=320)
bandwidth_extents = [10.0, 54.0]

# Class-related names. Be mindful of the ordering - positive, then negative
class_dirs = ['pos', 'neg']
section_suffixes = ['.song-section_S', '.nonsong-section_S']


# ----- Settings controlling training process --------------------------------
#
batch_size = 64
epochs = 60
epochs_between_evals = 5
buffer_size = 2048  # set high for good shuffling of training data
max_per_class_training_samples = 20000
validation_split = 0.15


# ----- LSTM related settings ------------------------------------------------
#
# Settings used during input-prep for LSTM training & inference
max_call_separation = 15 * 60  # 15 minutes
min_calls_in_song = 5

min_non_song_duration = 1.5 * 60  # 1.5 minutes

lstm_experiments = [
    # List of dicts with following fields
    #     segment_advance : In seconds.
    #     time_steps   : 108 for segment_advance of 1.00 s,
    #                    88 for segment_advance of 1.25 s
    #     pp           : Corresponds to 100%, 75%, 2/3 or 50% of time_steps
    #                    A setting of pp=0 effectively translates to a
    #                    prediction point at (time_steps - pp) = 100% of
    #                    input length.
    dict(segment_advance=1.00, time_steps=108, pp=0),
    dict(segment_advance=1.00, time_steps=108, pp=27),
    dict(segment_advance=1.00, time_steps=108, pp=36),
    dict(segment_advance=1.00, time_steps=108, pp=54),
    dict(segment_advance=1.25, time_steps=88, pp=0),
    dict(segment_advance=1.25, time_steps=88, pp=22),
    dict(segment_advance=1.25, time_steps=88, pp=29),
    dict(segment_advance=1.25, time_steps=88, pp=44),
]

# ============================================================================
