import os
import numpy as np
import matplotlib.pyplot as plt
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
        self.integer_array_fields = ["HOTA_TP", "HOTA_FN", "HOTA_FP"]
        self.float_array_fields = [
            "HOTA",
            "DetA",
            "AssA",
            "DetRe",
            "DetPr",
            "AssRe",
            "AssPr",
            "LocA",
            "OWTA",
        ]
        self.float_fields = ["HOTA(0)", "LocA(0)", "HOTALocA(0)", "TempLocAP"]
        self.fields = (
            self.float_array_fields + self.integer_array_fields + self.float_fields
        )
        self.summary_fields = self.float_array_fields + self.float_fields

    @_timing.time
    def eval_sequence(self, data):
        """Calculates the HOTA metrics for one sequence"""

        # Initialise results
        res = {}
        for field in self.float_array_fields + self.integer_array_fields:
            res[field] = np.zeros((len(self.array_labels)), dtype=float)
        for field in self.float_fields:
            res[field] = 0

        TempLocAP, _, _ = self.calculate_TempLocAP_merge(data)
        res["TempLocAP"] = TempLocAP

        # Return result quickly if tracker or gt sequence is empty
        if data["num_tracker_dets"] == 0:
            res["HOTA_FN"] = data["num_gt_dets"] * np.ones(
                (len(self.array_labels)), dtype=float
            )
            res["LocA"] = np.ones((len(self.array_labels)), dtype=float)
            res["LocA(0)"] = 1.0
            return res
        if data["num_gt_dets"] == 0:
            res["HOTA_FP"] = data["num_tracker_dets"] * np.ones(
                (len(self.array_labels)), dtype=float
            )
            res["LocA"] = np.ones((len(self.array_labels)), dtype=float)
            res["LocA(0)"] = 1.0
            return res

        # Variables counting global association
        potential_matches_count = np.zeros(
            (data["num_gt_ids"], data["num_tracker_ids"])
        )
        gt_id_count = np.zeros((data["num_gt_ids"], 1))
        tracker_id_count = np.zeros((1, data["num_tracker_ids"]))

        # First loop through each timestep and accumulate global track information.
        for t, (gt_ids_t, tracker_ids_t) in enumerate(
            zip(data["gt_ids"], data["tracker_ids"])
        ):
            # Count the potential matches between ids in each timestep
            # These are normalised, weighted by the match similarity.
            similarity = data["similarity_scores"][t]
            sim_iou_denom = (
                similarity.sum(0)[np.newaxis, :]
                + similarity.sum(1)[:, np.newaxis]
                - similarity
            )
            sim_iou = np.zeros_like(similarity)
            sim_iou_mask = sim_iou_denom > 0 + np.finfo("float").eps
            sim_iou[sim_iou_mask] = (
                similarity[sim_iou_mask] / sim_iou_denom[sim_iou_mask]
            )
            potential_matches_count[
                gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]
            ] += sim_iou

            # Calculate the total number of dets for each gt_id and tracker_id.
            gt_id_count[gt_ids_t] += 1
            tracker_id_count[0, tracker_ids_t] += 1

        # Calculate overall jaccard alignment score (before unique matching) between IDs
        global_alignment_score = potential_matches_count / (
            gt_id_count + tracker_id_count - potential_matches_count
        )
        matches_counts = [
            np.zeros_like(potential_matches_count) for _ in self.array_labels
        ]

        # Calculate scores for each timestep
        for t, (gt_ids_t, tracker_ids_t) in enumerate(
            zip(data["gt_ids"], data["tracker_ids"])
        ):
            # Deal with the case that there are no gt_det/tracker_det in a timestep.
            if len(gt_ids_t) == 0:
                for a, alpha in enumerate(self.array_labels):
                    res["HOTA_FP"][a] += len(tracker_ids_t)
                continue
            if len(tracker_ids_t) == 0:
                for a, alpha in enumerate(self.array_labels):
                    res["HOTA_FN"][a] += len(gt_ids_t)
                continue

            # Get matching scores between pairs of dets for optimizing HOTA
            similarity = data["similarity_scores"][t]
            score_mat = (
                global_alignment_score[
                    gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]
                ]
                * similarity
            )

            # Hungarian algorithm to find best matches
            match_rows, match_cols = linear_sum_assignment(-score_mat)

            # Calculate and accumulate basic statistics

            for a, alpha in enumerate(self.array_labels):
                actually_matched_mask = (
                    similarity[match_rows, match_cols] >= alpha - np.finfo("float").eps
                )
                alpha_match_rows = match_rows[actually_matched_mask]
                alpha_match_cols = match_cols[actually_matched_mask]
                num_matches = len(alpha_match_rows)
                res["HOTA_TP"][a] += num_matches
                res["HOTA_FN"][a] += len(gt_ids_t) - num_matches
                res["HOTA_FP"][a] += len(tracker_ids_t) - num_matches
                if num_matches > 0:
                    res["LocA"][a] += sum(
                        similarity[alpha_match_rows, alpha_match_cols]
                    )
                    matches_counts[a][
                        gt_ids_t[alpha_match_rows], tracker_ids_t[alpha_match_cols]
                    ] += 1

        # Calculate association scores (AssA, AssRe, AssPr) for the alpha value.
        # First calculate scores per gt_id/tracker_id combo and then average over the number of detections.
        for a, alpha in enumerate(self.array_labels):
            matches_count = matches_counts[a]
            ass_a = matches_count / np.maximum(
                1, gt_id_count + tracker_id_count - matches_count
            )
            res["AssA"][a] = np.sum(matches_count * ass_a) / np.maximum(
                1, res["HOTA_TP"][a]
            )
            ass_re = matches_count / np.maximum(1, gt_id_count)
            res["AssRe"][a] = np.sum(matches_count * ass_re) / np.maximum(
                1, res["HOTA_TP"][a]
            )
            ass_pr = matches_count / np.maximum(1, tracker_id_count)
            res["AssPr"][a] = np.sum(matches_count * ass_pr) / np.maximum(
                1, res["HOTA_TP"][a]
            )

        # Calculate final scores
        res["LocA"] = np.maximum(1e-10, res["LocA"]) / np.maximum(1e-10, res["HOTA_TP"])
        res = self._compute_final_fields(res)

        return res

    def calculate_TempLocAP(self, data):

        # Return result quickly if tracker or gt sequence is empty
        if data["num_tracker_dets"] == 0:
            if data["num_gt_dets"] == 0:
                precisions = np.array([1, 1])
                recalls = np.array([0, 1])
                TempLocAP = 1
                return TempLocAP, precisions, recalls
            else:
                precisions = np.array([0, 0])
                recalls = np.array([0, 1])
                TempLocAP = 0
                return TempLocAP, precisions, recalls
        if data["num_gt_dets"] == 0:
            precisions = np.array([0, 0])
            recalls = np.array([0, 1])
            TempLocAP = 0
            return TempLocAP, precisions, recalls

        TEMPORAL_IOU_THRESH = 0.5  # iou
        MATCHING_DIST_THRESH = 2.0  # m

        pred_tracks = {}
        gt_tracks = {}

        # Accumulate predicted and ground truth tracks from data
        for t in range(data["num_timesteps"]):
            for i, gt_id in enumerate(data["gt_ids"][t]):
                if gt_id not in gt_tracks:
                    gt_tracks[gt_id] = {}
                    gt_tracks[gt_id]["xy_pos"] = []
                    gt_tracks[gt_id]["timestamps"] = []
                    gt_tracks[gt_id]["category"] = data["gt_classes"][t][i]

                gt_tracks[gt_id]["xy_pos"].append(data["gt_dets"][t][i][:2])
                gt_tracks[gt_id]["timestamps"].append(t)

            for i, track_id in enumerate(data["tracker_ids"][t]):
                if track_id not in pred_tracks:
                    pred_tracks[track_id] = {}
                    pred_tracks[track_id]["confidence"] = data["tracker_confidences"][
                        t
                    ][i]
                    pred_tracks[track_id]["category"] = data["tracker_classes"][t][i]
                    pred_tracks[track_id]["xy_pos"] = []
                    pred_tracks[track_id]["timestamps"] = []

                pred_tracks[track_id]["xy_pos"].append(data["tracker_dets"][t][i][:2])
                pred_tracks[track_id]["timestamps"].append(t)

        # 1 to 1 match of predicted and ground truth tracks
        sorted_keys = sorted(
            pred_tracks.keys(),
            key=lambda key: pred_tracks[key]["confidence"],
            reverse=True,
        )

        # keys are track_ids, values are gt_ids
        matched_ids = {}

        # keys are track_ids, values are the iou of the timestamps of the matched predicted and ground truth timestamps
        matched_ious = {}
        unmatched_gt_ids = list(gt_tracks.keys())
        unmatched_track_ids = []

        for track_id in sorted_keys:
            track_stats = pred_tracks[track_id]

            track_traj = track_stats["xy_pos"]
            track_timestamps = track_stats["timestamps"]

            max_similarity = 0
            best_match = None
            corresponding_iou = 0
            for gt_id, gt_stats in gt_tracks.items():
                if (
                    gt_id not in unmatched_gt_ids
                    or gt_stats["category"] != track_stats["category"]
                ):
                    continue

                gt_traj = gt_stats["xy_pos"]
                gt_timestamps = gt_stats["timestamps"]

                intersection = len(
                    set(gt_timestamps).intersection((set(track_timestamps)))
                )
                union = len(set(gt_timestamps).union((set(track_timestamps))))
                iou = intersection / union

                if iou < TEMPORAL_IOU_THRESH:
                    continue

                total_distance = 0.0
                for timestamp in track_timestamps:
                    if timestamp in gt_timestamps:
                        total_distance += np.linalg.norm(
                            track_traj[track_timestamps.index(timestamp)]
                            - gt_traj[gt_timestamps.index(timestamp)]
                        )

                similarity_score = iou * max(
                    0.0,
                    1
                    - (
                        total_distance
                        / (
                            MATCHING_DIST_THRESH
                            * (intersection + np.finfo(np.float64).eps)
                        )
                    ),
                )
                if similarity_score > max_similarity:
                    max_similarity = similarity_score
                    best_match = gt_id

            if max_similarity > 0:
                matched_ids[track_id] = best_match
                matched_ious[track_id] = corresponding_iou
                unmatched_gt_ids.remove(best_match)
            else:
                unmatched_track_ids.append(track_id)

        # Compute precision and recall at all confidence thresholds.
        tp = np.zeros(len(pred_tracks))
        fp = np.zeros(len(pred_tracks))
        for i, (track_id, track_stats) in enumerate(pred_tracks.items()):

            if track_id in matched_ids:
                tp[i] = 1
            else:
                fp[i] = 1

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        recalls = tp / len(gt_tracks)
        precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

        assert np.all(0 <= precisions) & np.all(precisions <= 1)
        TempLocAP = self.get_ap(recalls, precisions)

        return TempLocAP, precisions, recalls

    def calculate_TempLocAP_concat(self, data):

        # Return result quickly if tracker or gt sequence is empty
        if data["num_tracker_dets"] == 0:
            if data["num_gt_dets"] == 0:
                precisions = np.array([1, 1])
                recalls = np.array([0, 1])
                return TempLocAP, precisions, recalls
            else:
                precisions = np.array([0, 0])
                recalls = np.array([0, 1])
                TempLocAP = 0
                return TempLocAP, precisions, recalls
        if data["num_gt_dets"] == 0:
            precisions = np.array([0, 0])
            recalls = np.array([0, 1])
            TempLocAP = 0
            return TempLocAP, precisions, recalls

        TEMPORAL_IOU_THRESH = 0.5  # iou
        MATCHING_DIST_THRESH = 2.0  # m

        pred_tracks = {}
        gt_tracks = {}

        # Accumulate predicted and ground truth tracks from data
        for t in range(data["num_timesteps"]):
            for i, gt_id in enumerate(data["gt_ids"][t]):
                if gt_id not in gt_tracks:
                    gt_tracks[gt_id] = {}
                    gt_tracks[gt_id]["xy_pos"] = []
                    gt_tracks[gt_id]["timestamps"] = []
                    gt_tracks[gt_id]["category"] = data["gt_classes"][t][i]

                gt_tracks[gt_id]["xy_pos"].append(data["gt_dets"][t][i][:2])
                gt_tracks[gt_id]["timestamps"].append(t)

            for i, track_id in enumerate(data["tracker_ids"][t]):
                if track_id not in pred_tracks:
                    pred_tracks[track_id] = {}
                    pred_tracks[track_id]["confidence"] = data["tracker_confidences"][
                        t
                    ][i]
                    pred_tracks[track_id]["category"] = data["tracker_classes"][t][i]
                    pred_tracks[track_id]["xy_pos"] = []
                    pred_tracks[track_id]["timestamps"] = []

                pred_tracks[track_id]["xy_pos"].append(data["tracker_dets"][t][i][:2])
                pred_tracks[track_id]["timestamps"].append(t)

        # 1 to 1 match of predicted and ground truth tracks
        pred_ids_by_conf = sorted(
            pred_tracks.keys(),
            key=lambda key: pred_tracks[key]["confidence"],
            reverse=True,
        )

        # Keys are match_id, values are dict of gt_id, pred_ids, gt_traj, pred_traj, category and confidence
        matched_predictions = {}

        # keys are track_ids, values are the iou of the timestamps of the matched predicted and ground truth timestamps
        unmatched_gt_ids = list(gt_tracks.keys())
        unmatched_track_ids = []

        for track_id in pred_ids_by_conf:
            track_stats = pred_tracks[track_id]
            track_confidence = track_stats["confidence"]
            track_traj = track_stats["xy_pos"]
            track_timestamps = track_stats["timestamps"]

            max_similarity = 0
            best_match_stats = None
            best_match = None
            for gt_id, gt_stats in gt_tracks.items():
                if (
                    gt_id not in unmatched_gt_ids
                    or gt_stats["category"] != track_stats["category"]
                ):
                    continue

                gt_traj = gt_stats["xy_pos"]
                gt_timestamps = gt_stats["timestamps"]

                intersection = len(
                    set(gt_timestamps).intersection((set(track_timestamps)))
                )
                union = len(set(gt_timestamps).union((set(track_timestamps))))
                iou = intersection / union

                if iou < TEMPORAL_IOU_THRESH:
                    continue

                distances = []
                for timestamp in track_timestamps:
                    if timestamp in gt_timestamps:
                        distances.append(
                            np.linalg.norm(
                                track_traj[track_timestamps.index(timestamp)]
                                - gt_traj[gt_timestamps.index(timestamp)]
                            )
                        )

                similarity_score = iou * max(
                    0, 1 - (np.mean(np.array(distances)) / MATCHING_DIST_THRESH)
                )

                if similarity_score > max_similarity:
                    max_similarity = similarity_score
                    best_match = gt_id
                    best_match_stats = {
                        "gt_id": gt_id,
                        "gt_timestamps": gt_timestamps,
                        "gt_traj": gt_traj,
                        "pred_ids": [track_id],
                        "pred_traj": track_traj,
                        "pred_timestamps": track_timestamps,
                        "category": gt_stats["category"],
                        "confidence": track_confidence,
                        "similarity": similarity_score,
                    }

            for match_id, match_stats in matched_predictions.items():

                gt_id = match_stats["gt_id"]
                gt_timestamps = match_stats["gt_timestamps"]
                gt_traj = match_stats["gt_traj"]

                pred_ids = match_stats["pred_ids"]
                pred_timestamps = match_stats["pred_timestamps"]
                pred_traj = match_stats["pred_traj"]

                category = match_stats["category"]
                confidence = match_stats["confidence"]
                match_similarity = match_stats["similarity"]

                if (
                    len(set(pred_timestamps).intersection(set(track_timestamps))) > 0
                    or track_stats["category"] != category
                ):
                    continue

                concat_pred_ids = pred_ids + track_id
                concat_timestamps = pred_timestamps + track_timestamps
                concat_traj = pred_traj + track_traj
                concat_confidence = confidence * len(
                    pred_timestamps
                ) + track_confidence * len(track_timestamps)
                concat_confidence /= len(concat_timestamps)

                intersection = len(
                    set(gt_timestamps).intersection(set(concat_timestamps))
                )
                union = len(set(gt_timestamps).union(set(concat_timestamps)))
                iou = intersection / union

                if iou < TEMPORAL_IOU_THRESH:
                    continue

                distances = []
                for timestamp in concat_timestamps:
                    if timestamp in gt_timestamps:
                        distances.append(
                            np.linalg.norm(
                                concat_traj[concat_timestamps.index(timestamp)]
                                - gt_traj[gt_timestamps.index(timestamp)]
                            )
                        )

                similarity_score = iou * max(
                    0, 1 - (np.mean(np.array(distances)) / MATCHING_DIST_THRESH)
                )

                if (
                    similarity_score > max_similarity
                    and similarity_score > match_similarity
                ):
                    max_similarity = similarity_score
                    best_match = match_id
                    best_match_stats = {
                        "pred_ids": concat_pred_ids,
                        "pred_traj": concat_traj,
                        "pred_timestamps": concat_timestamps,
                        "confidence": concat_confidence,
                        "similarity": similarity_score,
                    }

            if max_similarity > 0:
                if best_match < 0:
                    matched_predictions[best_match].update(best_match_stats)
                elif best_match is not None:
                    matched_predictions[-best_match - 1] = best_match_stats
                    unmatched_gt_ids.remove(best_match)
            else:
                unmatched_track_ids.append(track_id)

        concat_preds_by_conf = {}
        preds = list(matched_predictions.keys()) + unmatched_track_ids
        for pred in preds:
            if pred < 0:
                concat_preds_by_conf[pred] = matched_predictions[pred]["confidence"]
            else:
                concat_preds_by_conf[pred] = pred_tracks[pred]["confidence"]

        concat_ids_by_conf = sorted(
            concat_preds_by_conf.keys(),
            key=lambda key: concat_preds_by_conf[key],
            reverse=True,
        )

        # Compute precision and recall at all confidence thresholds.
        tp = np.zeros(len(concat_ids_by_conf))
        fp = np.zeros(len(concat_ids_by_conf))

        for i, concat_id in enumerate(concat_ids_by_conf):

            if concat_id < 0:
                tp[i] = 1
            else:
                fp[i] = 1

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        recalls = tp / len(gt_tracks)
        precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

        assert np.all(0 <= precisions) & np.all(precisions <= 1)
        TempLocAP = self.get_ap(recalls, precisions)

        return TempLocAP, precisions, recalls

    def calculate_TempLocAP_merge(self, data):

        # Return result quickly if tracker or gt sequence is empty
        if data["num_tracker_dets"] == 0:
            if data["num_gt_dets"] == 0:
                precisions = np.array([1, 1])
                recalls = np.array([0, 1])
                TempLocAP = 1
                return TempLocAP, precisions, recalls
            else:
                precisions = np.array([0, 0])
                recalls = np.array([0, 1])
                TempLocAP = 0
                return TempLocAP, precisions, recalls
        if data["num_gt_dets"] == 0:
            precisions = np.array([0, 0])
            recalls = np.array([0, 1])
            TempLocAP = 0
            return TempLocAP, precisions, recalls

        TEMPORAL_IOU_THRESH = 0.5  # iou
        MATCHING_DIST_THRESH = 2.0  # m

        pred_tracks = {}
        gt_tracks = {}

        # Accumulate predicted and ground truth tracks from data
        for t in range(data["num_timesteps"]):
            for i, gt_id in enumerate(data["gt_ids"][t]):
                if gt_id not in gt_tracks:
                    gt_tracks[gt_id] = {}
                    gt_tracks[gt_id]["xy_pos"] = []
                    gt_tracks[gt_id]["timestamps"] = []
                    gt_tracks[gt_id]["category"] = data["gt_classes"][t][i]

                gt_tracks[gt_id]["xy_pos"].append(data["gt_dets"][t][i][:2])
                gt_tracks[gt_id]["timestamps"].append(t)

            for i, track_id in enumerate(data["tracker_ids"][t]):
                if track_id not in pred_tracks:
                    pred_tracks[track_id] = {}
                    pred_tracks[track_id]["confidence"] = data["tracker_confidences"][
                        t
                    ][i]
                    pred_tracks[track_id]["category"] = data["tracker_classes"][t][i]
                    pred_tracks[track_id]["xy_pos"] = []
                    pred_tracks[track_id]["timestamps"] = []

                pred_tracks[track_id]["xy_pos"].append(data["tracker_dets"][t][i][:2])
                pred_tracks[track_id]["timestamps"].append(t)

        # 1 to 1 match of predicted and ground truth tracks
        pred_ids_by_conf = sorted(
            pred_tracks.keys(),
            key=lambda key: pred_tracks[key]["confidence"],
            reverse=True,
        )

        # keys are gt_id, values are list of corresponding pred_ids
        matched_ids = {}

        # keys are track_ids, values are the iou of the timestamps of the matched predicted and ground truth timestamps
        unmatched_track_ids = []

        for track_id in pred_ids_by_conf:
            track_stats = pred_tracks[track_id]
            track_confidence = track_stats["confidence"]
            track_traj = track_stats["xy_pos"]
            track_timestamps = track_stats["timestamps"]

            max_similarity = 0
            best_match = None
            for gt_id, gt_stats in gt_tracks.items():
                if gt_stats["category"] != track_stats["category"]:
                    continue

                gt_traj = gt_stats["xy_pos"]
                gt_timestamps = gt_stats["timestamps"]

                intersection = len(
                    set(gt_timestamps).intersection((set(track_timestamps)))
                )
                iol = intersection / len(track_timestamps)

                if iol < TEMPORAL_IOU_THRESH:
                    continue

                distances = []
                for timestamp in track_timestamps:
                    if timestamp in gt_timestamps:
                        distances.append(
                            np.linalg.norm(
                                track_traj[track_timestamps.index(timestamp)]
                                - gt_traj[gt_timestamps.index(timestamp)]
                            )
                        )

                similarity_score = iol * max(
                    0, 1 - (np.mean(np.array(distances)) / MATCHING_DIST_THRESH)
                )

                if similarity_score > max_similarity:
                    max_similarity = similarity_score
                    best_match = gt_id

            if max_similarity > 0:
                if best_match not in matched_ids:
                    matched_ids[best_match] = [track_id]
                else:
                    matched_ids[best_match].append(track_id)
            else:
                unmatched_track_ids.append(track_id)

        merged_predictions = {}
        for gt_id, pred_ids in matched_ids.items():
            merged_traj = []
            merged_timestamps = []
            merged_confidences = []
            merged_catetory = None

            for pred_id in pred_ids:
                track_timestamps = pred_tracks[pred_id]["timestamps"]
                track_trajectory = pred_tracks[pred_id]["xy_pos"]
                track_confidence = pred_tracks[pred_id]["confidence"]
                track_category = pred_tracks[pred_id]["category"]

                if len(merged_timestamps) == 0:
                    merged_timestamps.extend(track_timestamps)
                    merged_traj.extend(track_trajectory)
                    merged_catetory = track_category
                    merged_confidences.extend(
                        [track_confidence] * len(track_timestamps)
                    )
                    continue

                for i, timestamp in enumerate(track_timestamps):
                    if timestamp not in merged_timestamps:
                        insertion_index = 0
                        for merge_timestamp in merged_timestamps:
                            if merge_timestamp > timestamp:
                                insertion_index += 1

                        merged_timestamps.insert(insertion_index, timestamp)
                        merged_traj.insert(insertion_index, track_trajectory[i])
                        merged_confidences.insert(insertion_index, track_confidence)
                    else:
                        insertion_index = merged_timestamps.index(timestamp)
                        if track_confidence > merged_confidences[insertion_index]:
                            merged_confidences[insertion_index] = track_confidence
                            merged_traj[insertion_index] = track_trajectory[i]

            merged_predictions[-gt_id - 1] = {
                "xy_pos": merged_traj,
                "timestamps": merged_timestamps,
                "confidence": np.mean(np.array(merged_confidences)),
                "category": merged_catetory,
            }

        for unmatched_track_id in unmatched_track_ids:
            merged_predictions.update(
                {unmatched_track_id: pred_tracks[unmatched_track_id]}
            )

        merged_ids_by_conf = sorted(
            merged_predictions.keys(),
            key=lambda key: merged_predictions[key]["confidence"],
            reverse=True,
        )
        matched_gt_ids = []

        # Compute precision and recall at all confidence thresholds.
        tp = np.zeros(len(merged_ids_by_conf))
        fp = np.zeros(len(merged_ids_by_conf))

        for i, track_id in enumerate(merged_ids_by_conf):
            track_stats = merged_predictions[track_id]
            track_confidence = track_stats["confidence"]
            track_traj = track_stats["xy_pos"]
            track_timestamps = track_stats["timestamps"]

            max_similarity = 0
            best_match = None
            for gt_id, gt_stats in gt_tracks.items():
                if (
                    gt_id in matched_gt_ids
                    or gt_stats["category"] != track_stats["category"]
                ):
                    continue

                gt_traj = gt_stats["xy_pos"]
                gt_timestamps = gt_stats["timestamps"]

                intersection = len(
                    set(gt_timestamps).intersection((set(track_timestamps)))
                )
                union = len(set(gt_timestamps).union((set(track_timestamps))))
                iou = intersection / union

                if iou < TEMPORAL_IOU_THRESH:
                    continue

                distances = []
                for timestamp in track_timestamps:
                    if timestamp in gt_timestamps:
                        distances.append(
                            np.linalg.norm(
                                track_traj[track_timestamps.index(timestamp)]
                                - gt_traj[gt_timestamps.index(timestamp)]
                            )
                        )

                similarity_score = iou * max(
                    0, 1 - (np.mean(np.array(distances)) / MATCHING_DIST_THRESH)
                )

                if similarity_score > max_similarity:
                    max_similarity = similarity_score
                    best_match = gt_id

            if max_similarity > 0:
                matched_gt_ids.append(best_match)
                tp[i] = 1
            else:
                fp[i] = 1

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        recalls = tp / len(gt_tracks)
        precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

        assert np.all(0 <= precisions) & np.all(precisions <= 1)
        TempLocAP = self.get_ap(recalls, precisions)

        return TempLocAP, precisions, recalls

    def get_envelope(self, precisions):
        """Compute the precision envelope.

        Args:
        precisions:

        Returns:

        """
        for i in range(precisions.size - 1, 0, -1):
            precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
        return precisions

    def get_ap(self, recalls, precisions):
        """
        Calculate average precision.

        Args:
            recalls: Array of recall values
            precisions: Array of precision values

        Returns:
            float: average precision.
        """
        # first append sentinel values at the end
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))

        # get envelope (maximum precision for each recall value)
        precisions = self.get_envelope(precisions)

        # to calculate area under PR curve, look for points where X axis (recall) changes value
        i = np.where(recalls[1:] != recalls[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])

        return ap

    def plot_precision_recall_curve(
        self,
        recalls_list,
        precisions_list,
        ap_values=None,
        labels=None,
        colors=["blue", "green"],
        save_path=None,
    ):
        """
        Plot precision-recall curves for one or two sets of data.

        Args:
            recalls_list: List of recall arrays to plot
            precisions_list: List of precision arrays to plot
            ap_values: Optional list of AP values to display in the title
            labels: Optional list of labels for the legend
            colors: List of colors for the plots (default: blue and green)
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(8, 6))

        if not isinstance(recalls_list, list):
            recalls_list = [recalls_list]
        if not isinstance(precisions_list, list):
            precisions_list = [precisions_list]

        if labels is None:
            labels = [f"Curve {i+1}" for i in range(len(recalls_list))]

        for i, (recalls, precisions) in enumerate(zip(recalls_list, precisions_list)):
            color = colors[i % len(colors)]

            # Prepare data for plotting (add sentinel values)
            plot_recalls = np.concatenate(([0.0], recalls, [1.0]))
            plot_precisions = np.concatenate(([0.0], precisions, [0.0]))
            plot_precisions = self.get_envelope(plot_precisions.copy())

            # Plot the curve
            plt.plot(
                plot_recalls,
                plot_precisions,
                color=color,
                linestyle="-",
                linewidth=2,
                label=labels[i],
            )
            plt.fill_between(plot_recalls, 0, plot_precisions, alpha=0.1, color=color)

        # Set title
        if ap_values:
            ap_text = ", ".join(
                [f"{label}: AP = {ap:.3f}" for label, ap in zip(labels, ap_values)]
            )
            plt.title(f"Precision-Recall Curves ({ap_text})", fontsize=16)
        else:
            plt.title("Precision-Recall Curves", fontsize=16)

        # Set labels and limits
        plt.xlabel("Recall", fontsize=14)
        plt.ylabel("Precision", fontsize=14)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid(True)

        # Add legend if multiple curves
        if len(recalls_list) > 1:
            plt.legend(loc="lower left")

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for field in self.integer_array_fields:
            res[field] = self._combine_sum(all_res, field)
        for field in ["AssRe", "AssPr", "AssA"]:
            res[field] = self._combine_weighted_av(
                all_res, field, res, weight_field="HOTA_TP"
            )

        loca_weighted_sum = sum(
            [all_res[k]["LocA"] * all_res[k]["HOTA_TP"] for k in all_res.keys()]
        )
        res["LocA"] = np.maximum(1e-10, loca_weighted_sum) / np.maximum(
            1e-10, res["HOTA_TP"]
        )

        tlap_weighted_sum = sum(
            [
                all_res[k]["TempLocAP"]
                * (all_res[k]["HOTA_TP"][0] + all_res[k]["HOTA_FN"][0])
                for k in all_res.keys()
            ]
        )
        res["TempLocAP"] = np.maximum(1e-10, tlap_weighted_sum) / np.maximum(
            1e-10, res["HOTA_TP"][0] + res["HOTA_FN"][0]
        )

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
                    {
                        k: v
                        for k, v in all_res.items()
                        if (
                            v["HOTA_TP"] + v["HOTA_FN"] + v["HOTA_FP"]
                            > 0 + np.finfo("float").eps
                        ).any()
                    },
                    field,
                )
            else:
                res[field] = self._combine_sum(
                    {k: v for k, v in all_res.items()}, field
                )

        for field in self.float_fields + self.float_array_fields:
            if ignore_empty_classes:
                res[field] = np.mean(
                    [
                        v[field]
                        for v in all_res.values()
                        if (
                            v["HOTA_TP"] + v["HOTA_FN"] + v["HOTA_FP"]
                            > 0 + np.finfo("float").eps
                        ).any()
                    ],
                    axis=0,
                )
            else:
                res[field] = np.mean([v[field] for v in all_res.values()], axis=0)
        return res

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        res = {}
        for field in self.integer_array_fields:
            res[field] = self._combine_sum(all_res, field)
        for field in ["AssRe", "AssPr", "AssA"]:
            res[field] = self._combine_weighted_av(
                all_res, field, res, weight_field="HOTA_TP"
            )

        loca_weighted_sum = sum(
            [all_res[k]["LocA"] * all_res[k]["HOTA_TP"] for k in all_res.keys()]
        )
        res["LocA"] = np.maximum(1e-10, loca_weighted_sum) / np.maximum(
            1e-10, res["HOTA_TP"]
        )

        tlap_weighted_sum = sum(
            [
                all_res[k]["TempLocAP"]
                * (all_res[k]["HOTA_TP"][0] + all_res[k]["HOTA_FN"][0])
                for k in all_res.keys()
            ]
        )
        res["TempLocAP"] = np.maximum(1e-10, tlap_weighted_sum) / np.maximum(
            1e-10, res["HOTA_TP"][0] + res["HOTA_FN"][0]
        )

        res = self._compute_final_fields(res)
        return res

    def _compute_final_fields(self, res):
        """Calculate sub-metric ('field') values which only depend on other sub-metric values.
        This function is used both for both per-sequence calculation, and in combining values across sequences.
        """
        res["DetRe"] = res["HOTA_TP"] / np.maximum(1, res["HOTA_TP"] + res["HOTA_FN"])
        res["DetPr"] = res["HOTA_TP"] / np.maximum(1, res["HOTA_TP"] + res["HOTA_FP"])
        res["DetA"] = res["HOTA_TP"] / np.maximum(
            1, res["HOTA_TP"] + res["HOTA_FN"] + res["HOTA_FP"]
        )
        res["HOTA"] = np.sqrt(res["DetA"] * res["AssA"])
        res["OWTA"] = np.sqrt(res["DetRe"] * res["AssA"])

        res["HOTA(0)"] = res["HOTA"][0]
        res["LocA(0)"] = res["LocA"][0]
        res["HOTALocA(0)"] = res["HOTA(0)"] * res["LocA(0)"]
        return res

    def plot_single_tracker_results(self, table_res, tracker, cls, output_folder):
        """Create plot of results"""

        # Only loaded when run to reduce minimum requirements
        from matplotlib import pyplot as plt

        res = table_res["COMBINED_SEQ"]
        styles_to_plot = ["r", "b", "g", "b--", "b:", "g--", "g:", "m", "o--", "o:"]
        for name, style in zip(self.float_array_fields, styles_to_plot):
            plt.plot(self.array_labels, res[name], style)
        plt.xlabel("alpha")
        plt.ylabel("score")
        plt.title(tracker + " - " + cls)
        plt.axis([0, 1, 0, 1])
        legend = []
        for name in self.float_array_fields:
            legend += [name + " (" + str(np.round(np.mean(res[name]), 2)) + ")"]
        plt.legend(legend, loc="lower left")
        out_file = os.path.join(output_folder, cls + "_plot.pdf")
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        plt.savefig(out_file)
        plt.savefig(out_file.replace(".pdf", ".png"))
        plt.clf()
