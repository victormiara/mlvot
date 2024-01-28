import numpy as np
import cv2
import os
from scipy.optimize import linear_sum_assignment


def load_detections(file_path):
    return np.loadtxt(file_path, delimiter=',', usecols=range(10))

def calculate_iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    inter_y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)
    union_area = box1[2] * box1[3] + box2[2] * box2[3] - inter_area
    return inter_area / union_area if union_area else 0

def update_tracks_with_hungarian(detections, tracks, frame_number, iou_threshold=0.5):
    if not tracks:
        for det in detections:
            if int(det[0]) == frame_number:
                tracks.append({'bbox': det[2:6], 'track_id': len(tracks), 'last_updated': frame_number})
        return tracks
    
    cost_matrix = np.zeros((len(tracks), len(detections)))
    for t_idx, track in enumerate(tracks):
        for d_idx, det in enumerate(detections):
            if int(det[0]) == frame_number:
                cost_matrix[t_idx, d_idx] = -calculate_iou(track['bbox'], det[2:6])
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    for t_idx, d_idx in zip(row_ind, col_ind):
        if -cost_matrix[t_idx, d_idx] > iou_threshold:
            tracks[t_idx]['bbox'] = detections[d_idx][2:6]
            tracks[t_idx]['last_updated'] = frame_number
        else:
            tracks[t_idx]['last_updated'] = -1

    tracks = [track for track in tracks if track['last_updated'] == frame_number]

    unmatched_detections = [d_idx for d_idx in range(len(detections)) if d_idx not in col_ind]
    for d_idx in unmatched_detections:
        det = detections[d_idx]
        if int(det[0]) == frame_number:
            tracks.append({'bbox': det[2:6], 'track_id': len(tracks), 'last_updated': frame_number})

    return tracks


def draw_tracking_results(frame, tracks):
    for track in tracks:
        cv2.rectangle(frame, (int(track['bbox'][0]), int(track['bbox'][1])), (int(track['bbox'][0] + track['bbox'][2]), int(track['bbox'][1] + track['bbox'][3])), (255, 0, 0), 2)
        cv2.putText(frame, f"ID: {track['track_id']}", (int(track['bbox'][0]), int(track['bbox'][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

detections = load_detections('../ADL-Rundle-6/det/det.txt')
tracks = []
sigma_iou = 0.5

frames_path = '../ADL-Rundle-6/img1/'
total_frames = 500
all_tracking_data = []

for frame_number in range(1, total_frames + 1):
    frame_path = os.path.join(frames_path, f"{frame_number:06d}.jpg")
    frame = cv2.imread(frame_path)
    if frame is None:
        break

    current_frame_detections = np.array([d for d in detections if int(d[0]) == frame_number])
    tracks = update_tracks_with_hungarian(current_frame_detections, tracks, frame_number, sigma_iou)
    current_frame_tracks = [
        {'frame_number': frame_number, 'track': track} 
        for track in tracks if track['last_updated'] == frame_number
    ]
    all_tracking_data.extend(current_frame_tracks)
    frame_with_tracking = draw_tracking_results(frame, tracks)
    cv2.imshow('Tracking', frame_with_tracking)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

def save_tracking_results(tracking_data, sequence_name):
    with open(f"{sequence_name}.txt", 'w') as f_out:
        for item in tracking_data:
            frame_number = item['frame_number']
            track = item['track']
            bbox = track['bbox']
            line = f"{frame_number},{track['track_id']},{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])},1,-1,-1,-1\n"
            f_out.write(line)

save_tracking_results(all_tracking_data, 'ADL-Rundle-6')