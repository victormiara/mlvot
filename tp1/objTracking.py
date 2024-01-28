import cv2
from KalmanFilter import KalmanFilter
from Detector import detect  # Assuming detect() is the function for object detection


kf = KalmanFilter(dt=0.1, u_x=1, u_y=1, std_acc=1, x_sdt_means=0.1, y_sdt_means=0.1)

cap = cv2.VideoCapture('randomball.avi')  # Replace 'path_to_video' with the actual video path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Object Detection
    centroid = detect(frame)  # Assuming detect() returns the centroid of the detected object

    # Kalman Filter Prediction and Update
    if centroid is not None:
        kf.predict()
        kf.update(centroid)

        # Visualization
        # Draw detected circle (green)
        cv2.circle(frame, tuple(centroid), radius=5, color=(0, 255, 0), thickness=-1)

        # Draw predicted position (blue rectangle)
        predicted_position = (int(kf.X[0]), int(kf.X[1]))
        cv2.rectangle(frame, predicted_position, predicted_position, color=(255, 0, 0), thickness=2)

        # Draw estimated position (red rectangle)
        estimated_position = (int(kf.X[0] + kf.X[2]), int(kf.X[1] + kf.X[3]))
        cv2.rectangle(frame, estimated_position, estimated_position, color=(0, 0, 255), thickness=2)

        # Draw trajectory (tracking path)
        # This part requires keeping track of past positions to draw the path

    cv2.imshow('Object Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()