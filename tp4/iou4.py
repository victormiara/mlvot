import numpy as np
import cv2
import os
from scipy.optimize import linear_sum_assignment

class KalmanFilter:
    def __init__(self, dt=0.1, u_x=1, u_y=1, std_acc=1, x_dt_means=0.1, y_dt_means=0.1) -> None:
        self.dt = dt
        self.u = np.matrix([[u_x], [u_y]]) 
        self.std_acc = std_acc
        self.x_dt_means = x_dt_means
        self.y_dt_means = y_dt_means
        self.state_matrix = np.zeros((4, 1))
        self.A = np.matrix([[1, 0, dt, 0],
                            [0, 1, 0, dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        self.B = np.matrix([[(dt**2)/2, 0],
                            [0, (dt**2)/2],
                            [dt, 0],
                            [0, dt]])
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        self.Q = np.matrix([[(dt**4)/4, 0, (dt**3)/2, 0],
                            [0, (dt**4)/4, 0, (dt**3)/2],
                            [(dt**3)/2, 0, dt**2, 0],
                            [0, (dt**3)/2, 0, dt**2]]) * self.std_acc**2
        self.R = np.matrix([[x_dt_means**2, 0],
                            [0, y_dt_means**2]])
        
        self.P = np.eye(self.A.shape[1])

    def predict(self):
        self.state_matrix = np.dot(self.A, self.state_matrix) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.state_matrix
    
    def update(self, Z):
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        Z = np.matrix([[self.x_dt_means], [self.y_dt_means]])
        y = Z - np.dot(self.H, self.state_matrix)
        self.state_matrix = self.state_matrix + np.dot(K, y)
        self.P = (np.eye(self.H.shape[1]) - (K * self.H)) * self.P
        return self.state_matrix


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
    for track in tracks:
        track['predicted_state'] = track['kalman_filter'].predict()

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for t_idx, track in enumerate(tracks):
        for d_idx, det in enumerate(detections):
            if int(det[0]) == frame_number:
                predicted_bbox = [track['predicted_state'][0], track['predicted_state'][1], track['bbox'][2], track['bbox'][3]]
                iou = calculate_iou(predicted_bbox, det[2:6])
                cost_matrix[t_idx, d_idx] = -iou

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_detections = []
    for t_idx, d_idx in zip(row_ind, col_ind):
        if -cost_matrix[t_idx, d_idx] > iou_threshold:
            tracks[t_idx]['bbox'] = detections[d_idx][2:6]
            tracks[t_idx]['last_updated'] = frame_number
            tracks[t_idx]['kalman_filter'].update(detections[d_idx][2:4])
            matched_detections.append(d_idx)

    for d_idx, det in enumerate(detections):
        if d_idx not in matched_detections and int(det[0]) == frame_number:
            kf = KalmanFilter()
            kf.state_matrix[:2] = np.matrix(det[2:4]).T
            new_track = {
                'bbox': det[2:6],
                'track_id': len(tracks) + 1,
                'kalman_filter': kf,
                'last_updated': frame_number
            }
            tracks.append(new_track)

    tracks = [track for track in tracks if track['last_updated'] == frame_number]

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