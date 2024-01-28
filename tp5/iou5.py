import numpy as np
import cv2
import os
from scipy.optimize import linear_sum_assignment
import torch
from torchvision.models import resnet50
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from torchvision.models.feature_extraction import create_feature_extractor

model = resnet50(pretrained=True)
model.eval()

preprocess = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def create_resnet_feature_extractor(model):
    return create_feature_extractor(model, return_nodes={'avgpool': 'pool'})

feature_extractor = create_resnet_feature_extractor(model)

def extract_features(image_patch):
    if image_patch.size == 0 or image_patch.shape[0] == 0 or image_patch.shape[1] == 0:
        return torch.zeros([2048])
    
    if image_patch.size == 0 or image_patch.shape[0] == 0 or image_patch.shape[1] == 0:
        return torch.zeros([2048])
    
    image_tensor = preprocess(Image.fromarray(image_patch))
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(image_tensor)['pool']
    features = torch.flatten(features, 1)
    return features.squeeze(0)

def similarity_score(features1, features2):
    return torch.nn.functional.cosine_similarity(features1, features2, dim=0)

def get_image_patch(frame, bbox):
    return frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

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

def update_tracks_with_hungarian(detections, tracks, frame, frame_number, iou_threshold=0.5, iou_weight=0.5, feature_weight=0.5, max_lost_frames=10, detection_confidence_threshold=0.5):
    detections_dicts = []
    for det in detections:
        if int(det[0]) == frame_number:
            bbox = [int(det[2]), int(det[3]), int(det[4]), int(det[5])]
            image_patch = get_image_patch(frame, bbox)
            det_features = extract_features(image_patch)
            det_confidence = det[6] if len(det) > 6 else 1.0
            detections_dicts.append({
                'frame': int(det[0]),
                'bbox': bbox,
                'features': det_features,
                'detection_confidence': det_confidence
            })

    for track in tracks:
        track['predicted_state'] = track['kalman_filter'].predict()

    cost_matrix = np.zeros((len(tracks), len(detections_dicts)), dtype=np.float32)
    for t_idx, track in enumerate(tracks):
        for d_idx, det in enumerate(detections_dicts):
            iou_score = calculate_iou(track['bbox'], det['bbox'])
            feature_similarity = similarity_score(track['features'], det['features']).item()
            cost = -iou_weight * iou_score - feature_weight * (1 - feature_similarity)
            cost_matrix[t_idx, d_idx] = cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_detections = []
    for t_idx, d_idx in zip(row_ind, col_ind):
        if -cost_matrix[t_idx, d_idx] > iou_threshold:
            tracks[t_idx]['bbox'] = detections_dicts[d_idx]['bbox']
            tracks[t_idx]['last_updated'] = frame_number
            tracks[t_idx]['kalman_filter'].update(np.matrix(detections_dicts[d_idx]['bbox'][:2]).T)
            matched_detections.append(d_idx)

    # Track management
    unmatched_tracks = set(range(len(tracks))) - set(row_ind)
    unmatched_detections = set(range(len(detections_dicts))) - set(col_ind)

    for d_idx in unmatched_detections:
        det = detections_dicts[d_idx]
        if det['detection_confidence'] > detection_confidence_threshold:
            kf = KalmanFilter()
            kf.state_matrix[:2] = np.matrix(det['bbox'][:2]).T
            new_track = {
                'bbox': det['bbox'],
                'track_id': len(tracks) + 1,
                'kalman_filter': kf,
                'last_updated': frame_number,
                'lost_frames': 0,
                'features': det['features'],
                'detection_confidence': det['detection_confidence']
            }
            tracks.append(new_track)

    # Terminate old tracks or update lost frame count
    for t_idx in unmatched_tracks:
        track = tracks[t_idx]
        track['lost_frames'] += 1
        if track['lost_frames'] > max_lost_frames:
            tracks[t_idx]['terminate'] = True  # Mark track for termination

    # Remove terminated tracks
    tracks = [track for track in tracks if not track.get('terminate', False)]

    return tracks

def draw_tracking_results(frame, tracks):
    for track in tracks:
        cv2.rectangle(frame, (int(track['bbox'][0]), int(track['bbox'][1])), (int(track['bbox'][0] + track['bbox'][2]), int(track['bbox'][1] + track['bbox'][3])), (255, 0, 0), 2)
        cv2.putText(frame, f"ID: {track['track_id']}", (int(track['bbox'][0]), int(track['bbox'][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

detections = load_detections('../ADL-Rundle-6/det/det.txt')
tracks = []

frames_path = '../ADL-Rundle-6/img1/'
total_frames = 500
all_tracking_data = []

for frame_number in range(1, total_frames + 1):
    frame_path = os.path.join(frames_path, f"{frame_number:06d}.jpg")
    frame = cv2.imread(frame_path)
    if frame is None:
        break

    current_frame_detections = np.array([d for d in detections if int(d[0]) == frame_number])
    
    for track in tracks:
        if 'features' not in track:
            bbox = track['bbox']
            image_patch = get_image_patch(frame, [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
            track['features'] = extract_features(image_patch)
    
    tracks = update_tracks_with_hungarian(current_frame_detections, tracks, frame, frame_number)
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