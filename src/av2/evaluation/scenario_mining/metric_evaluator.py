import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import copy
import tkinter as tk
from tkinter import messagebox

class TempLocAPCalculator:
    def __init__(self):
        pass
        
    def get_ap(self, recalls, precisions):
        # Standard VOC AP calculation
        # First append sentinel values at the end
        recalls = np.concatenate(([0.], recalls, [1.]))
        precisions = np.concatenate(([0.], precisions, [0.]))

        # Compute the precision envelope
        for i in range(precisions.size - 1, 0, -1):
            precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

        # Compute area under PR curve
        indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
        ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
        return ap
        
    def calculate_TempLocAP(self, data):
        # Return result quickly if tracker or gt sequence is empty
        if data['num_tracker_dets'] == 0:
            if data['num_gt_dets'] == 0:
                precisions = np.array([1, 1])
                recalls = np.array([0, 1])
                TempLocAP = 1
                return TempLocAP, precisions, recalls
            else:
                precisions = np.array([0, 0])
                recalls = np.array([0, 1])
                TempLocAP = 0
                return TempLocAP, precisions, recalls
        if data['num_gt_dets'] == 0:
            precisions = np.array([0, 0])
            recalls = np.array([0, 1])
            TempLocAP = 0
            return TempLocAP, precisions, recalls

        TEMPORAL_IOU_THRESH  = 0.5 #iou
        MATCHING_DIST_THRESH = 2.0 #m

        pred_tracks = {}
        gt_tracks = {}

        #Accumulate predicted and ground truth tracks from data
        for t in range(data['num_timesteps']):
            for i, gt_id in enumerate(data['gt_ids'][t]):
                if gt_id not in gt_tracks:
                    gt_tracks[gt_id] = {}
                    gt_tracks[gt_id]['xy_pos'] = []
                    gt_tracks[gt_id]['timestamps'] = []
                    gt_tracks[gt_id]['category'] = data['gt_classes'][t][i]

                gt_tracks[gt_id]['xy_pos'].append(data['gt_dets'][t][i][:2])
                gt_tracks[gt_id]['timestamps'].append(t)

            for i, track_id in enumerate(data['tracker_ids'][t]):
                if track_id not in pred_tracks:
                    pred_tracks[track_id] = {}
                    pred_tracks[track_id]['confidence'] = data['tracker_confidences'][t][i]
                    pred_tracks[track_id]['category'] = data['tracker_classes'][t][i]
                    pred_tracks[track_id]['xy_pos'] = []
                    pred_tracks[track_id]['timestamps'] = []

                pred_tracks[track_id]['xy_pos'].append(data['tracker_dets'][t][i][:2])
                pred_tracks[track_id]['timestamps'].append(t)

        # 1 to 1 match of predicted and ground truth tracks
        sorted_keys = sorted(pred_tracks.keys(), key=lambda key: pred_tracks[key]['confidence'], reverse=True)

        #keys are track_ids, values are gt_ids
        matched_ids = {} 

        #keys are track_ids, values are the iou of the timestamps of the matched predicted and ground truth timestamps
        matched_ious = {} 
        unmatched_gt_ids = list(gt_tracks.keys())
        unmatched_track_ids = []

        for track_id in sorted_keys:
            track_stats = pred_tracks[track_id]

            track_traj = track_stats['xy_pos']
            track_timestamps = track_stats['timestamps']
            
            max_similarity = 0
            best_match = None
            corresponding_iou = 0
            for gt_id, gt_stats in gt_tracks.items():
                if gt_id not in unmatched_gt_ids or gt_stats['category'] != track_stats['category']:
                    continue

                gt_traj = gt_stats['xy_pos']
                gt_timestamps = gt_stats['timestamps']

                intersection = len(set(gt_timestamps).intersection((set(track_timestamps))))
                union = len(set(gt_timestamps).union((set(track_timestamps))))
                iou = intersection/union

                total_distance = 0.0
                for timestamp in track_timestamps:
                    if timestamp in gt_timestamps:
                        total_distance += np.linalg.norm(
                            track_traj[track_timestamps.index(timestamp)] - gt_traj[gt_timestamps.index(timestamp)])


                similarity_score = iou * max(0.0, 
                    1 - (total_distance/(MATCHING_DIST_THRESH*(intersection+np.finfo(np.float64).eps))))
                if similarity_score > max_similarity:
                    max_similarity = similarity_score
                    best_match = gt_id
                    corresponding_iou = iou

            if max_similarity > 0:
                matched_ids[track_id] = best_match
                matched_ious[track_id] = corresponding_iou
                unmatched_gt_ids.remove(best_match)
            else:
                unmatched_track_ids.append(track_id)

        #Compute precision and recall at all confidence thresholds.
        tp = np.zeros(len(pred_tracks))
        fp = np.zeros(len(pred_tracks))
        for i, (track_id, track_stats) in enumerate(pred_tracks.items()):

            if track_id in matched_ids and matched_ious[track_id] >= TEMPORAL_IOU_THRESH:
                tp[i] = 1
            else:
                fp[i] = 1

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        recalls = tp/max(len(gt_tracks), 1)
        precisions = tp/np.maximum(tp+fp, np.finfo(np.float64).eps)

        assert np.all(0 <= precisions) & np.all(precisions <= 1)
        TempLocAP = self.get_ap(recalls, precisions)

        return TempLocAP, precisions, recalls


class TrackDrawUI:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.25)
        
        # Configure the plot
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 10)
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Space")
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Track data
        self.tracks = {
            'gt': [],      # Ground truth tracks (orange)
            'pred': []     # Prediction tracks (green with varying opacity)
        }
        
        # Drawing mode and state
        self.mode = 'gt'  # Initial mode: ground truth
        self.drawing = False
        self.start_point = None
        self.current_line = None
        self.current_confidence = 1.0  # Default confidence for predictions
        
        # Setup UI elements
        axcolor = 'lightgoldenrodyellow'
        
        # Mode buttons
        self.ax_mode_gt = plt.axes([0.15, 0.15, 0.15, 0.05])
        self.ax_mode_pred = plt.axes([0.15, 0.05, 0.15, 0.05])
        self.btn_mode_gt = Button(self.ax_mode_gt, 'Ground Truth', color='orange', hovercolor='darkorange')
        self.btn_mode_pred = Button(self.ax_mode_pred, 'Prediction', color='lightgreen', hovercolor='green')
        
        # Confidence slider for predictions
        self.ax_confidence = plt.axes([0.35, 0.10, 0.35, 0.03], facecolor=axcolor)
        self.slider_confidence = Slider(self.ax_confidence, 'Confidence', 0.05, 1.0, valinit=1.0, valstep=0.05)
        
        # Action buttons
        self.ax_clear = plt.axes([0.75, 0.15, 0.1, 0.05])
        self.ax_calculate = plt.axes([0.75, 0.05, 0.1, 0.05])
        self.btn_clear = Button(self.ax_clear, 'Clear', color='lightcoral', hovercolor='red')
        self.btn_calculate = Button(self.ax_calculate, 'Calculate', color='lightblue', hovercolor='blue')
        
        # Connect events
        self.btn_mode_gt.on_clicked(self.set_mode_gt)
        self.btn_mode_pred.on_clicked(self.set_mode_pred)
        self.slider_confidence.on_changed(self.update_confidence)
        self.btn_clear.on_clicked(self.clear_tracks)
        self.btn_calculate.on_clicked(self.calculate_metric)
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        # Legend
        self.update_legend()
        
        # Calculator for the metric
        self.calculator = TempLocAPCalculator()
        
        plt.show()
    
    def set_mode_gt(self, event):
        self.mode = 'gt'
        print(f"Mode: Ground Truth")
    
    def set_mode_pred(self, event):
        self.mode = 'pred'
        print(f"Mode: Prediction (Confidence: {self.current_confidence:.1f})")
    
    def update_confidence(self, val):
        self.current_confidence = val
        print(f"Confidence set to: {self.current_confidence:.1f}")
    
    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        
        self.drawing = True
        self.start_point = (event.xdata, event.ydata)
        
        # Create a temporary line
        if self.mode == 'gt':
            self.current_line, = self.ax.plot([event.xdata], [event.ydata], 'o-', color='orange', linewidth=2)
        else:  # pred mode
            self.current_line, = self.ax.plot([event.xdata], [event.ydata], 'o-', color='green', 
                                             alpha=self.current_confidence, linewidth=2)
        
        self.fig.canvas.draw()
    
    def on_motion(self, event):
        if not self.drawing or event.inaxes != self.ax or self.current_line is None:
            return
        
        # Update temporary line
        x_vals = [self.start_point[0], event.xdata]
        y_vals = [self.start_point[1], event.ydata]
        self.current_line.set_data(x_vals, y_vals)
        self.fig.canvas.draw()
    
    def on_release(self, event):
        if not self.drawing or self.start_point is None or self.current_line is None:
            return
        
        self.drawing = False
        end_point = (event.xdata, event.ydata)
        
        # Finalize the line if it was drawn in the axes
        if event.inaxes == self.ax:
            track = {
                'start': self.start_point,
                'end': end_point,
                'line': self.current_line
            }
            
            if self.mode == 'pred':
                track['confidence'] = self.current_confidence
                
                # Add confidence text label next to the line
                mid_x = (self.start_point[0] + end_point[0]) / 2
                mid_y = (self.start_point[1] + end_point[1]) / 2
                # Offset the text slightly from the line
                offset = .66
                if end_point[1] > self.start_point[1]:
                    offset_y = offset
                else:
                    offset_y = -offset
                
                conf_text = self.ax.text(mid_x, mid_y + offset_y, f"Conf: {self.current_confidence:.2f}", 
                                         color='darkgreen', fontweight='bold', 
                                         ha='center', va='center',
                                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='green', boxstyle='round,pad=0.3'))
                track['conf_text'] = conf_text
            
            self.tracks[self.mode].append(track)
            self.current_line = None
            
            # Update legend to reflect the new track
            self.update_legend()
        else:
            # If released outside axes, remove temporary line
            self.current_line.remove()
            self.current_line = None
        
        self.fig.canvas.draw()
    
    def clear_tracks(self, event):
        # Remove all lines from the plot
        for track_type in self.tracks:
            for track in self.tracks[track_type]:
                track['line'].remove()
                
                # Also remove confidence text labels for prediction tracks
                if track_type == 'pred' and 'conf_text' in track:
                    track['conf_text'].remove()
        
        # Clear the track lists
        self.tracks = {'gt': [], 'pred': []}
        
        # Update legend
        self.update_legend()
        self.fig.canvas.draw()
        print("All tracks cleared")
    
    def update_legend(self):
        # Create legend elements
        legend_elements = [
            Line2D([0], [0], color='orange', lw=2, label=f'Ground Truth ({len(self.tracks["gt"])})'),
            Line2D([0], [0], color='green', lw=2, label=f'Prediction ({len(self.tracks["pred"])})')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')
    
    def interpolate_track(self, start, end, num_steps):
        """Interpolate points along a track."""
        t_start, s_start = start
        t_end, s_end = end
        
        t_vals = np.linspace(t_start, t_end, num_steps)
        s_vals = np.linspace(s_start, s_end, num_steps)
        
        return t_vals, s_vals
    
    def create_metric_data(self):
        """Convert drawn tracks to the format expected by the TempLocAP calculator."""
        # Define time resolution
        time_min = 0
        time_max = 100
        num_timesteps = 100
        timesteps = np.linspace(time_min, time_max, num_timesteps).astype(int)
        
        # Initialize data structure
        data = {
            'num_timesteps': num_timesteps,
            'gt_ids': [[] for _ in range(num_timesteps)],
            'gt_dets': [[] for _ in range(num_timesteps)],
            'gt_classes': [[] for _ in range(num_timesteps)],
            'tracker_ids': [[] for _ in range(num_timesteps)],
            'tracker_dets': [[] for _ in range(num_timesteps)],
            'tracker_classes': [[] for _ in range(num_timesteps)],
            'tracker_confidences': [[] for _ in range(num_timesteps)],
            'num_gt_dets': 0,
            'num_tracker_dets': 0,
        }
        
        # Process ground truth tracks
        for i, track in enumerate(self.tracks['gt']):
            gt_id = i + 1  # Track ID (starting from 1)
            start, end = track['start'], track['end']
            
            # Interpolate track
            t_vals, s_vals = self.interpolate_track(start, end, 50)
            
            for t, s in zip(t_vals, s_vals):
                # Find appropriate timestep
                t_idx = min(int(t), num_timesteps - 1)
                
                # Add detection to appropriate timestep
                data['gt_ids'][t_idx].append(gt_id)
                data['gt_dets'][t_idx].append(np.array([s, 0]))  # Using 1D space, setting y=0
                data['gt_classes'][t_idx].append(1)  # All tracks have the same class
                data['num_gt_dets'] += 1
        
        # Process prediction tracks
        for i, track in enumerate(self.tracks['pred']):
            track_id = i + 1  # Track ID (starting from 1)
            start, end = track['start'], track['end']
            confidence = track.get('confidence', 1.0)
            
            # Interpolate track
            t_vals, s_vals = self.interpolate_track(start, end, 50)
            
            for t, s in zip(t_vals, s_vals):
                # Find appropriate timestep
                t_idx = min(int(t), num_timesteps - 1)
                
                # Add detection to appropriate timestep
                data['tracker_ids'][t_idx].append(track_id)
                data['tracker_dets'][t_idx].append(np.array([s, 0]))  # Using 1D space, setting y=0
                data['tracker_classes'][t_idx].append(1)  # All tracks have the same class
                data['tracker_confidences'][t_idx].append(confidence)
                data['num_tracker_dets'] += 1
        
        return data
    
    def calculate_metric(self, event):
        if not self.tracks['gt'] and not self.tracks['pred']:
            messagebox.showinfo("Empty Tracks", "Please draw at least one track before calculating.")
            return
        
        # Create data for the calculator
        data = self.create_metric_data()
        
        # Calculate the metric
        TempLocAP, precisions, recalls = self.calculator.calculate_TempLocAP(data)
        
        # Display results
        result_text = f"TempLocAP: {TempLocAP:.4f}"
        print(result_text)
        
        # Create a plot window for precision-recall curve with area highlighted
        fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
        
        # Sort the recalls and corresponding precisions
        sorted_indices = np.argsort(recalls)
        sorted_recalls = recalls[sorted_indices]
        sorted_precisions = precisions[sorted_indices]
        
        # Add a point at recall=0 if not already present
        if sorted_recalls[0] != 0:
            sorted_recalls = np.concatenate(([0], sorted_recalls))
            sorted_precisions = np.concatenate(([sorted_precisions[0]], sorted_precisions))
        
        # Add a point at recall=1 if not already present
        if sorted_recalls[-1] != 1:
            sorted_recalls = np.concatenate((sorted_recalls, [1]))
            sorted_precisions = np.concatenate((sorted_precisions, [0]))  # Precision typically drops to 0 at recall=1
            
        # Plot precision-recall curve
        ax_pr.plot(sorted_recalls, sorted_precisions, '-o', color='blue', label='Precision-Recall Curve')
        
        # Fill the area under the curve
        ax_pr.fill_between(sorted_recalls, sorted_precisions, alpha=0.3, color='skyblue', label=f'AP = {TempLocAP:.4f}')
        
        # Add explanatory annotation about the area
        ax_pr.annotate(f'Area = {TempLocAP:.4f}', 
                     xy=(0.5, 0.5), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="orange", alpha=0.8),
                     ha='center', fontsize=12)
        
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_xlim([0, 1])
        ax_pr.set_ylim([0, 1.05])
        ax_pr.set_title(f'Precision-Recall Curve\nTempLocAP = {TempLocAP:.4f}')
        ax_pr.grid(True)
        ax_pr.legend(loc='lower left')
        plt.tight_layout()
        plt.show()
        
        # Show a message box with the score
        messagebox.showinfo("TempLocAP Result", result_text)


# Run the application
if __name__ == "__main__":
    ui = TrackDrawUI()