import numpy as np

class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_sdt_means, y_sdt_means):
        # Time step
        self.dt = dt

        # Control input (acceleration)
        self.u = np.array([[u_x], [u_y]])

        # Initial state matrix
        self.X = np.array([[0], [0], [0], [0]])  # x0, y0, vx, vy

        # System model matrices
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        self.B = np.array([[0.5 * dt**2, 0],
                           [0, 0.5 * dt**2],
                           [dt, 0],
                           [0, dt]])

        # meansurement mapping matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # Process noise covariance
        self.Q = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]]) * std_acc**2

        # meansurement noise covariance
        self.R = np.array([[x_sdt_means**2, 0],
                           [0, y_sdt_means**2]])

        # Prediction error covariance
        self.P = np.eye(4)
        
    def predict(self):
        # Predict the next state
        self.X = np.dot(self.A, self.X) + np.dot(self.B, self.u)

        # Predict the error covariance
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        return self.X
    
    def update(self, z):
        # meansurement residual
        y = z - np.dot(self.H, self.X)

        # Residual covariance
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Kalman gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Update the state estimate
        self.X += np.dot(K, y)

        # Update the error covariance
        I = np.eye(self.H.shape[1])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

        return self.X