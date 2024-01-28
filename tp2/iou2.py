import numpy as np
import cv2
import os

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

def greedy_associate(detections, tracks, frame_number, sigma_iou=0.5):
    matched_detections = set()
    for track in tracks:
        best_iou = sigma_iou
        best_detection_index = None
        for d_idx, detection in enumerate(detections):
            if d_idx in matched_detections:
                continue
            iou = calculate_iou(track['bbox'], detection[2:6])
            if iou > best_iou:
                best_iou = iou
                best_detection_index = d_idx
        if best_detection_index is not None:
            track['bbox'] = detections[best_detection_index][2:6]
            track['last_updated'] = frame_number
            matched_detections.add(best_detection_index)
        else:
            track['last_updated'] = -1

    tracks = [track for track in tracks if track['last_updated'] == frame_number]

    for d_idx, detection in enumerate(detections):
        if d_idx not in matched_detections:
            tracks.append({'bbox': detection[2:6], 'track_id': len(tracks), 'last_updated': frame_number})

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

for frame_number in range(1, total_frames + 1):
    frame_path = os.path.join(frames_path, f"{frame_number:06d}.jpg")
    frame = cv2.imread(frame_path)
    if frame is None:
        break

    current_frame_detections = np.array([d for d in detections if int(d[0]) == frame_number])
    tracks = greedy_associate(current_frame_detections, tracks, frame_number, sigma_iou)
    frame_with_tracking = draw_tracking_results(frame, tracks)
    cv2.imshow('Tracking', frame_with_tracking)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
