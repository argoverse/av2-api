import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from trackeval.metrics._base_metric import _BaseMetric
from trackeval import _timing


class HOTA(_BaseMetric):
    """Class which implements the HOTA metrics.
    See: https://link.springer.com/article/10.1007/s11263-020-01375-2
    """

    def __init__(self, config=None):
        super().__init__()
        self.plottable = True
        self.array_labels = np.arange(0.05, 0.99, 0.05)
        self.integer_array_fields = ['HOTA_TP', 'HOTA_FN', 'HOTA_FP']
        self.float_array_fields = ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA', 'TempLocPr', 'TempLocRe', 'OWTA']
        self.float_fields = ['HOTA(0)', 'LocA(0)', 'HOTALocA(0)', 'TempLocAP']
        self.fields = self.float_array_fields + self.integer_array_fields + self.float_fields
        self.summary_fields = self.float_array_fields + self.float_fields

    @_timing.time
    def eval_sequence(self, data):
        """Calculates the HOTA metrics for one sequence"""

        # Initialise results
        res = {}
        for field in self.float_array_fields + self.integer_array_fields:
            res[field] = np.zeros((len(self.array_labels)), dtype=np.float64)
        for field in self.float_fields:
            res[field] = 0

        # Return result quickly if tracker or gt sequence is empty
        if data['num_tracker_dets'] == 0:
            res['HOTA_FN'] = data['num_gt_dets'] * np.ones((len(self.array_labels)), dtype=np.float64)
            res['LocA'] = np.ones((len(self.array_labels)), dtype=np.float64)
            res['LocA(0)'] = 1.0

            if data['num_gt_dets'] == 0:
                res['TempLocAP'] = 1
            else:
                res['TempLocAP'] = 0
            return res
        if data['num_gt_dets'] == 0:
            res['HOTA_FP'] = data['num_tracker_dets'] * np.ones((len(self.array_labels)), dtype=np.float64)
            res['LocA'] = np.ones((len(self.array_labels)), dtype=np.float64)
            res['LocA(0)'] = 1.0
            res['TempLocAP'] = 0
            return res

        # Variables counting global association
        potential_matches_count = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
        gt_id_count = np.zeros((data['num_gt_ids'], 1))
        tracker_id_count = np.zeros((1, data['num_tracker_ids']))
        
        gt_ids = set()
        tracker_ids = set()
        # First loop through each timestep and accumulate global track information.
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            # Count the potential matches between ids in each timestep
            # These are normalised, weighted by the match similarity.
            similarity = data['similarity_scores'][t]
            sim_iou_denom = similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
            sim_iou = np.zeros_like(similarity)
            sim_iou_mask = sim_iou_denom > 0 + np.finfo('float').eps
            sim_iou[sim_iou_mask] = similarity[sim_iou_mask] / sim_iou_denom[sim_iou_mask]
            potential_matches_count[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] += sim_iou
            
            gt_ids = gt_ids.union(set(gt_ids_t))
            tracker_ids = tracker_ids.union(set(tracker_ids_t))

            # Calculate the total number of dets for each gt_id and tracker_id.
            gt_id_count[gt_ids_t] += 1
            tracker_id_count[0, tracker_ids_t] += 1

        # Calculate overall jaccard alignment score (before unique matching) between IDs
        global_alignment_score = potential_matches_count / (gt_id_count + tracker_id_count - potential_matches_count)
        matches_counts = [np.zeros_like(potential_matches_count) for _ in self.array_labels]

        #Initializing matches dict where keys are gt_ids and values are a list of corresponding pred_ids
        matches = {gt_id: set() for gt_id in gt_ids}

        # Calculate scores for each timestep
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            # Deal with the case that there are no gt_det/tracker_det in a timestep.
            if len(gt_ids_t) == 0:
                for a, alpha in enumerate(self.array_labels):
                    res['HOTA_FP'][a] += len(tracker_ids_t)
                continue
            if len(tracker_ids_t) == 0:
                for a, alpha in enumerate(self.array_labels):
                    res['HOTA_FN'][a] += len(gt_ids_t)
                continue

            # Get matching scores between pairs of dets for optimizing HOTA
            similarity = data['similarity_scores'][t]
            score_mat = global_alignment_score[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] * similarity

            # Hungarian algorithm to find best matches
            match_rows, match_cols = linear_sum_assignment(-score_mat)

            for i in range(len(match_rows)):
                matches[data['gt_ids'][t][match_rows[i]]].add(data['tracker_ids'][t][match_cols[i]])

            # Calculate and accumulate basic statistics
            for a, alpha in enumerate(self.array_labels):
                actually_matched_mask = similarity[match_rows, match_cols] >= alpha - np.finfo('float').eps
                alpha_match_rows = match_rows[actually_matched_mask]
                alpha_match_cols = match_cols[actually_matched_mask]
                num_matches = len(alpha_match_rows)
                res['HOTA_TP'][a] += num_matches
                res['HOTA_FN'][a] += len(gt_ids_t) - num_matches
                res['HOTA_FP'][a] += len(tracker_ids_t) - num_matches
                if num_matches > 0:
                    res['LocA'][a] += sum(similarity[alpha_match_rows, alpha_match_cols])
                    matches_counts[a][gt_ids_t[alpha_match_rows], tracker_ids_t[alpha_match_cols]] += 1

        gt_time_segs = {gt_id: {'timestamps': set()} for gt_id in gt_ids}

        for t in range(len(data['gt_ids'])):
            for gt_id in gt_ids:
                if gt_id in data['gt_ids'][t]:
                    gt_time_segs[gt_id]['timestamps'].add(t)

        unionized_tracker_ids = set()
        for track_ids in matches.values():
            unionized_tracker_ids = unionized_tracker_ids.union(track_ids)

        unmatched_tracker_ids = tracker_ids.difference(unionized_tracker_ids)
        track_time_segs = {tracker_id: {'timestamps': set(), 'confidence': 0} for tracker_id in unmatched_tracker_ids}

        for t in range(len(data['tracker_ids'])):
            for tracker_id in tracker_ids:
                if tracker_id in data['tracker_ids'][t] and tracker_id not in unionized_tracker_ids:
                    track_time_segs[tracker_id]['timestamps'].add(t)

                    id_index = np.where(data['tracker_ids'][t] == tracker_id)[0][0]
                    track_time_segs[tracker_id]['confidence'] = data['tracker_confidences'][t][id_index]

        for gt_id, track_ids in matches.items():
            if not track_ids:
                continue

            track_time_segs[-gt_id-1] = {}
            track_time_segs[-gt_id-1]['timestamps'] = set()
            track_time_segs[-gt_id-1]['confidence'] = 0

            confidences = []
            for t in range(len(data['gt_ids'])):
                for track_id in track_ids:
                    if track_id in data['tracker_ids'][t]:
                        id_index = np.where(data['tracker_ids'][t] == track_id)[0][0]
                        confidences.append(data['tracker_confidences'][t][id_index])
                        track_time_segs[-gt_id-1]['timestamps'].add(t)
            track_time_segs[-gt_id-1]['confidence'] = np.mean(np.array(confidences))
        
        for a, alpha in enumerate(self.array_labels):
            tp = 0
            fn = 0
            fp = 0

            for gt_id in gt_ids:
                if (len(matches[gt_id]) == 0 
                or track_time_segs[-gt_id-1]['confidence'] < alpha):
                    fn += 1
                    continue
                
                intersection = gt_time_segs[gt_id]['timestamps'].intersection(track_time_segs[-gt_id-1]['timestamps'])
                union = gt_time_segs[gt_id]['timestamps'].union(track_time_segs[-gt_id-1]['timestamps'])
                iou = len(intersection)/len(union)

                if iou >= 0.5:
                    tp += 1
                else: 
                    fn += 1

            for tracker_id, stats in track_time_segs.items():
                if tracker_id >= 0 and stats['confidence'] > alpha:
                    fp += 1

            if tp+fp == 0:
                res['TempLocPr'][a] = 0
            else:
                res['TempLocPr'][a] = tp / (tp+fp)

            if tp+fn == 0:
                res['TempLocRe'][a] = 0
            else:
                res['TempLocRe'][a] = tp / (tp+fn)

        # Calculate association scores (AssA, AssRe, AssPr) for the alpha value.
        # First calculate scores per gt_id/tracker_id combo and then average over the number of detections.
        for a, alpha in enumerate(self.array_labels):
            matches_count = matches_counts[a]
            ass_a = matches_count / np.maximum(1, gt_id_count + tracker_id_count - matches_count)
            res['AssA'][a] = np.sum(matches_count * ass_a) / np.maximum(1, res['HOTA_TP'][a])
            ass_re = matches_count / np.maximum(1, gt_id_count)
            res['AssRe'][a] = np.sum(matches_count * ass_re) / np.maximum(1, res['HOTA_TP'][a])
            ass_pr = matches_count / np.maximum(1, tracker_id_count)
            res['AssPr'][a] = np.sum(matches_count * ass_pr) / np.maximum(1, res['HOTA_TP'][a])

        # Calculate final scores
        res['LocA'] = np.maximum(1e-10, res['LocA']) / np.maximum(1e-10, res['HOTA_TP'])
        res = self._compute_final_fields(res)
        return res

    
    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for field in self.integer_array_fields:
            res[field] = self._combine_sum(all_res, field)
        for field in ['AssRe', 'AssPr', 'AssA', 'TempLocPr', 'TempLocRe']:
            res[field] = self._combine_weighted_av(all_res, field, res, weight_field='HOTA_TP')

        loca_weighted_sum = sum([all_res[k]['LocA'] * all_res[k]['HOTA_TP'] for k in all_res.keys()])
        res['LocA'] = np.maximum(1e-10, loca_weighted_sum) / np.maximum(1e-10, res['HOTA_TP'])
        res = self._compute_final_fields(res)
        return res

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=False):
        """Combines metrics across all classes by averaging over the class values.
        If 'ignore_empty_classes' is True, then it only sums over classes with at least one gt or predicted detection.
        """
        res = {}
        for field in self.integer_array_fields:
            if ignore_empty_classes:
                res[field] = self._combine_sum(
                    {k: v for k, v in all_res.items()
                     if (v['HOTA_TP'] + v['HOTA_FN'] + v['HOTA_FP'] > 0 + np.finfo('float').eps).any()}, field)
            else:
                res[field] = self._combine_sum({k: v for k, v in all_res.items()}, field)

        for field in self.float_fields + self.float_array_fields:
            if ignore_empty_classes:
                res[field] = np.mean([v[field] for v in all_res.values() if
                                      (v['HOTA_TP'] + v['HOTA_FN'] + v['HOTA_FP'] > 0 + np.finfo('float').eps).any()],
                                     axis=0)
            else:
                res[field] = np.mean([v[field] for v in all_res.values()], axis=0)
        return res

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        res = {}
        for field in self.integer_array_fields:
            res[field] = self._combine_sum(all_res, field)
        for field in ['AssRe', 'AssPr', 'AssA', 'TempLocRe', 'TempLocPr']:
            res[field] = self._combine_weighted_av(all_res, field, res, weight_field='HOTA_TP')


        loca_weighted_sum = sum([all_res[k]['LocA'] * all_res[k]['HOTA_TP'] for k in all_res.keys()])
        res['LocA'] = np.maximum(1e-10, loca_weighted_sum) / np.maximum(1e-10, res['HOTA_TP'])
        res = self._compute_final_fields(res)
        return res

    def _compute_final_fields(self, res):
        """Calculate sub-metric ('field') values which only depend on other sub-metric values.
        This function is used both for both per-sequence calculation, and in combining values across sequences.
        """
        res['DetRe'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FN'])
        res['DetPr'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FP'])
        res['DetA'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FN'] + res['HOTA_FP'])
        res['HOTA'] = np.sqrt(res['DetA'] * res['AssA'])
        res['OWTA'] = np.sqrt(res['DetRe'] * res['AssA'])

        res['TempLocAP'] = self.compute_average_precision(res['TempLocPr'], res['TempLocRe'])
        res['HOTA(0)'] = res['HOTA'][0]
        res['LocA(0)'] = res['LocA'][0]
        res['HOTALocA(0)'] = res['HOTA(0)']*res['LocA(0)']
        return res
    
    @staticmethod
    def compute_average_precision(precisions, recalls):
        """
        Compute Average Precision using numpy's trapz function.
        
        Args:
            precisions: List of precision values
            recalls: List of recall values
        
        Returns:
            Average Precision value
        """
        
        # Sort by recall
        sorted_indices = np.argsort(recalls)
        recalls = np.array(recalls)[sorted_indices]
        precisions = np.array(precisions)[sorted_indices]
        
        # Interpolate precision values
        for i in range(len(precisions)-2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i+1])
        
        # Compute AP using trapezoid rule
        ap = np.trapezoid(y=precisions, x=recalls)
        
        return ap

    def plot_single_tracker_results(self, table_res, tracker, cls, output_folder):
        """Create plot of results"""

        # Only loaded when run to reduce minimum requirements
        from matplotlib import pyplot as plt

        res = table_res['COMBINED_SEQ']
        styles_to_plot = ['r', 'b', 'g', 'b--', 'b:', 'g--', 'g:', 'm', 'o--', 'o:']
        for name, style in zip(self.float_array_fields, styles_to_plot):
            plt.plot(self.array_labels, res[name], style)
        plt.xlabel('alpha')
        plt.ylabel('score')
        plt.title(tracker + ' - ' + cls)
        plt.axis([0, 1, 0, 1])
        legend = []
        for name in self.float_array_fields:
            legend += [name + ' (' + str(np.round(np.mean(res[name]), 2)) + ')']
        plt.legend(legend, loc='lower left')
        out_file = os.path.join(output_folder, cls + '_plot.pdf')
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        plt.savefig(out_file)
        plt.savefig(out_file.replace('.pdf', '.png'))
        plt.clf()