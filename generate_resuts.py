import os
import numpy as np
import json
from matplotlib import pyplot as plt

from temporalcontext import settings
from temporalcontext.functions import read_selmap, read_folds_info, \
    gen_list_of_section_files, LSTMData, chunk_up, match_segments_to_selections


scores_filename_fmt = 'scores_TS{:d}_PP{:d}.json'
model_types = ['cnn', 't1', 't2', 't3']
type_specs = dict(
    #    name, color
    cnn=('CNN',     '#7f7f7f'),
    t1 =('$LSTM_{score}$', '#E76D54'),
    t2 =('$LSTM_{feature}$', '#AE017E'),
    t3 =('$LSTM_{s+f}$', '#2171B5')
)

thld_pts = np.round(np.linspace(0.0, 1.0, 41).astype(np.float32), 3)
coverage_frac_thld = 0.75

selmap = read_selmap(os.path.join(settings.raw_data_root, 'selmap.csv'))
fold_file_idxs = read_folds_info(os.path.join(settings.raw_data_root, 'folds_info.txt'))


def plot_perf_result_figs(clip_advance, ts, pp):

    num_folds = len(fold_file_idxs)

    # Convert to start-end pairs
    thld_edges = np.stack([thld_pts[:-1], thld_pts[1:]], axis=1)
    # Add the last point itself as a bin
    thld_edges = np.concatenate([thld_edges, [[thld_pts[-1], np.inf]]], axis=0)

    # Result containers
    counts_for_pr = {
        m_type: {
            'num_tps_per_thld_bin': np.zeros((num_folds, thld_edges.shape[0]), dtype=np.uint64),
            'num_dets_per_thld_bin': np.zeros((num_folds, thld_edges.shape[0]), dtype=np.uint64),
            'num_recalls_above_thld': np.zeros((num_folds, thld_edges.shape[0]), dtype=np.uint64)
        } for m_type in model_types
    }
    fold_gt_selections = np.zeros((num_folds, ), dtype=np.uint64)
    fold_added_annotations = np.zeros((num_folds, ), dtype=np.uint64)
    # Counts, for calibration
    pi_0_num, pi_0_denom = 0, 0
    fold_pi_num, fold_pi_denom = np.zeros((num_folds, ), dtype=np.uint64), np.zeros((num_folds, ), dtype=np.uint64)

    for fold_idx, fold_info in enumerate(fold_file_idxs):

        fold_seg_root = os.path.join(settings.project_root, settings.folds_dir,
                                     'f{:02d}'.format(fold_idx + 1),
                                     'seg_adv_{:.2f}'.format(clip_advance))
        input_root = os.path.join(fold_seg_root, settings.lstm_data_dir)
        scores_root = os.path.join(fold_seg_root, settings.scores_dir)

        fold_res = json.load(open(os.path.join(scores_root, scores_filename_fmt.format(ts, pp)), 'r'))
        fold_seg_scores = fold_res['scores']
        fold_score_ranges = fold_res['score_ranges']

        # Process test files corresponding to the current fold
        for sec_file in gen_list_of_section_files(
                input_root,
                [selmap[f_idx][0] for f_idx in fold_info['test']],
                settings.section_suffixes):

            # Load data
            seg_start, seg_end, t1_input, _, _, selections = LSTMData.read(sec_file, ts, pp)

            if t1_input.shape[0] < ts:
                continue

            # Clip start & end times for each Y
            _, clip_start_ends = chunk_up(t1_input, np.arange(t1_input.shape[0]), ts, pp)
            clip_start_ends = seg_start + (pp * clip_advance) + (clip_start_ends * clip_advance)
            clip_start_ends = np.stack([clip_start_ends, clip_start_ends + settings.segment_length], axis=1)

            # Only 'starts' were made available. Add 'ends' and make an Nx2 array.
            selections = np.stack([selections, selections + settings.annot_duration], axis=1)

            # Keep only those selections that occur within "valid" Ys' temporal extremes
            selections = selections[np.logical_and(selections[:, 0] >= clip_start_ends[0, 0],
                                                   selections[:, 1] <= clip_start_ends[-1, 1]), ...]

            # Load secondary annotations
            added_annots_fullpath = os.path.join(settings.raw_data_root, settings.added_annot_dir, sec_file[len(input_root)+1:])
            if os.path.exists(added_annots_fullpath):
                with np.load(added_annots_fullpath) as d:
                    added_annots = d['begin_time']
                # Only 'starts' were loaded. Add 'ends' and make an Nx2 array.
                added_annots = np.stack([added_annots, added_annots + settings.annot_duration], axis=1)

                # Keep only those added annots that occur within "valid" Ys' temporal extremes
                added_annots = added_annots[np.logical_and(added_annots[:, 0] >= clip_start_ends[0, 0],
                                                           added_annots[:, 1] <= clip_start_ends[-1, 1]), ...]
                fold_added_annotations[fold_idx] += added_annots.shape[0]

                selections = np.concatenate([selections, added_annots])

            fold_gt_selections[fold_idx] += selections.shape[0]

            # Get coverage mask for clips overlapping with selections
            ground_truth = match_segments_to_selections(clip_start_ends, selections, coverage_frac_thld)

            fold_pi_num[fold_idx] += ground_truth.sum()
            fold_pi_denom[fold_idx] += len(ground_truth)
            pi_0_num += ground_truth.sum()
            pi_0_denom += len(ground_truth)

            # Gather stats
            for m_type in model_types:

                # Model's predictions for the current seg_file
                m_res = np.asarray(fold_seg_scores[sec_file[len(input_root) + 1:]][m_type], dtype=np.float32)

                # Normalize the scores to fall within the min-max for the current fold
                m_res = ((m_res - fold_score_ranges[m_type][0]) /
                         (fold_score_ranges[m_type][1] - fold_score_ranges[m_type][0]))

                # Perform matching with original selections and accumulate counts
                num_tps_per_thld_bin, num_dets_per_thld_bin, num_recalls_over_thld = \
                    assess_dets_with_coverage(m_res, ground_truth, clip_start_ends, selections,
                                              coverage_frac_thld, thld_edges)
                counts_for_pr[m_type]['num_tps_per_thld_bin'][fold_idx, :] += num_tps_per_thld_bin
                counts_for_pr[m_type]['num_dets_per_thld_bin'][fold_idx, :] += num_dets_per_thld_bin
                counts_for_pr[m_type]['num_recalls_above_thld'][fold_idx, :] += num_recalls_over_thld

    # ========== All counts are now accumulated. Compute PRs ==========

    pi_0 = float(pi_0_num) / float(pi_0_denom)
    fold_pis = fold_pi_num.astype(np.float32) / fold_pi_denom.astype(np.float32)

    fold_calibration_ratios = fold_pis * (1 - pi_0) / (pi_0 * (1 - fold_pis))

    aggregates = dict()
    for m_type in model_types:
        prec, reca = calculate_folds_prs(counts_for_pr[m_type], fold_calibration_ratios, fold_gt_selections)
        aggregates[m_type] = dict(precision=nan_median(prec), recall=nan_median(reca))

    # ========== All necessary items are now computed. Start plotting ==========

    plot_aggregates(aggregates, thld_edges)

    # Save the fig
    result_name = 'segadv{:.2f}s_PP{:d}'.format(clip_advance, pp)
    plt.savefig(os.path.join(settings.project_root, settings.perf_results_dir, result_name + '_result.png'),
                dpi=200, bbox_inches='tight')


def assess_dets_with_coverage(scores, ground_truth, clip_start_ends, selections, overlap_thld_for_R, thld_edges):

    dets_in_thld_bins_mask = np.stack([
        np.logical_and(scores >= bin_l, scores < bin_u)
        for bin_l, bin_u in thld_edges
    ])

    num_dets_per_thld_bin = dets_in_thld_bins_mask.sum(axis=1).astype(np.uint64)  # num dets (TP + FP) per thld bin

    if selections.shape[0] > 0:

        # num TPs in each thld bin
        num_tps_per_thld_bin = np.logical_and(dets_in_thld_bins_mask, [ground_truth]).sum(axis=1)

        clip_annot_overlaps = np.stack([
            (np.minimum(clip_e, selections[:, 1]) - np.maximum(clip_s, selections[:, 0]))
            for clip_s, clip_e in clip_start_ends])

        clip_top_match_sel_idxs = np.argmax(clip_annot_overlaps, axis=1)
        clip_sel_min_coverage = np.asarray([clip_annot_overlaps[c_idx, s_idx] >= (settings.annot_duration * overlap_thld_for_R)
                                            for c_idx, s_idx in enumerate(clip_top_match_sel_idxs)])

        dets_above_thld_mask = np.stack([
            (scores >= bin_l)
            for bin_l in thld_edges[:, 0]
        ])
        tps_above_thld_mask = np.logical_and(dets_above_thld_mask, [clip_sel_min_coverage])

        recalled_selections_above_thld = np.asarray([
            np.unique(clip_top_match_sel_idxs[tps_above_thld]).shape[0]
            for tps_above_thld in tps_above_thld_mask])  # num GT selections that were recalled

    else:
        num_tps_per_thld_bin = np.zeros_like(num_dets_per_thld_bin)
        recalled_selections_above_thld = np.zeros_like(num_dets_per_thld_bin)

    return num_tps_per_thld_bin.astype(np.uint64), \
           num_dets_per_thld_bin.astype(np.uint64), \
           recalled_selections_above_thld.astype(np.uint64)


def calculate_folds_prs(counts_dict, calibration_ratios, fold_num_gts):

    num_tps_per_thld_bin = counts_dict['num_tps_per_thld_bin'].astype(np.float32)
    num_dets_per_thld_bin = counts_dict['num_dets_per_thld_bin'].astype(np.float32)
    num_recalls_above_thld = counts_dict['num_recalls_above_thld'].astype(np.float32)

    num_tps_above_thld = np.stack([  # num TPs above each thld
        num_tps_per_thld_bin[:, idx:].sum(axis=1) for idx in range(num_tps_per_thld_bin.shape[1])], axis=1)
    num_dets_above_thld = np.stack([  # num dets above each thld
        num_dets_per_thld_bin[:, idx:].sum(axis=1) for idx in range(num_dets_per_thld_bin.shape[1])], axis=1)

    precision = np.full_like(num_tps_per_thld_bin, np.nan, dtype=np.float32)
    prec_denom = num_tps_above_thld + (
            np.expand_dims(calibration_ratios, axis=1) * (num_dets_above_thld - num_tps_above_thld))
    valid_mask = prec_denom > 0.0
    precision[valid_mask] = num_tps_above_thld[valid_mask] / prec_denom[valid_mask]

    recall = num_recalls_above_thld / np.expand_dims(fold_num_gts.astype(np.float32), 1)

    return precision, recall


def plot_aggregates(aggregates, thld_edges):

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw={'wspace': 0.2}, figsize=(6, 2.6))
    ax1.set_title('Precision-Recall Curve')
    ax2.set_title('F1-score vs. Threshold')

    for m_type in aggregates.keys():

        f1_score_vals = ((2 * aggregates[m_type]['precision'] * aggregates[m_type]['recall']) /
                         (aggregates[m_type]['precision'] + aggregates[m_type]['recall']))
        max_f1_idx = np.argmax(f1_score_vals)

        name, color = type_specs[m_type]

        # Add median PRs
        ax1.plot(aggregates[m_type]['recall'], aggregates[m_type]['precision'],
                 linewidth=1.35, label=name, color=color)
        ax1.plot([aggregates[m_type]['recall'][max_f1_idx]], [aggregates[m_type]['precision'][max_f1_idx]],
                 marker='D', color=color)

        # Add F1-score vs. threshold
        ax2.plot(thld_edges[:, 0], f1_score_vals,
                 linewidth=1.35, label=name, color=color)
        ax2.plot([thld_edges[max_f1_idx, 0]], [f1_score_vals[max_f1_idx]],
                 marker='D', color=color)

    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.0])
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax2.set_xlabel('Normalized Threshold')
    ax2.set_ylabel('F1-score')
    ax1.grid('both')
    ax2.grid('both')
    ax1.legend(loc='best')

    return fig


def nan_median(data):

    non_nan_thld_mask = np.logical_not(np.all(np.isnan(data), axis=0))

    ret_m = np.full((data.shape[1], ), np.nan, dtype=data.dtype)

    ret_m[non_nan_thld_mask] = np.nanmedian(data[:, non_nan_thld_mask], axis=0)

    return ret_m


if __name__ == '__main__':

    os.makedirs(os.path.join(settings.project_root, settings.perf_results_dir), exist_ok=True)

    for lstm_exp in settings.lstm_experiments:
        print('# ---------- TS={:3d}, PP={:d} ----------'.format(lstm_exp['time_steps'], lstm_exp['pp']))

        plot_perf_result_figs(lstm_exp['segment_advance'], lstm_exp['time_steps'], lstm_exp['pp'])

