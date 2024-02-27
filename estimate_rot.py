import numpy as np
from scipy import io
from quaternion import Quaternion   # numpy-quaternio
import math


import matplotlib.pyplot as plt





#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an unscented kalman filter


# def skew(v):
#     x, y, z = v
#     return np.array([[0, -z, y],
#                      [z, 0, -x],
#                      [-y, x, 0]])

def unskew(m):
    return np.array([[(m[2][1] - m[1][2])/2, (m[0][2] - m[2][0])/2, (m[1][0] - m[0][1])/2]])


def calibrate_accelerometer(accel, rots, vicon, time):
    
    T = time.shape[0] + 1

    # Transpose to fit my math better:w
    ARaw_B = accel.T
    # Append constant term to introduce bias in the math
    ones_column = np.ones((ARaw_B.shape[0], 1))
    ARaw_B = np.hstack((ARaw_B, ones_column))




    ## Assemble the vector of rotation matrices

    # Transpose each 3x3 (by swapping the first 2 dims)
    R_WB_transpose = np.transpose(rots, axes=(1, 0, 2))
    # Pad each 3x3 rot matrix into a 4x4
    # Specifically, add to the start and end of each dimension (0,1), (0,1), (0,1) many constant_values
    R_WB_transpose = np.pad(R_WB_transpose, ((0, 1), (0, 1), (0, 0)), mode='constant', constant_values=0)
    # Set bottom right term of each homogenized rotation matrix to 1
    R_WB_transpose[3, 3, :] = 1 
    # But actually numpy's batch dimension should be the first for both things
    R_WB_transpose = np.transpose(R_WB_transpose, axes=(2, 0, 1))





    # Assemble the vector of ground truth a_W (just gravity) A_W
    A_W = np.zeros((T-1, 4, 1))
    A_W[:, 2, 0] = 9.81
    A_W[:, 3, 0] = 1


    
    ## Solve the linear regression problem
    
    # Create Y and X based upon our problem's variables
    Y = np.matmul(R_WB_transpose, A_W)
    Y = np.squeeze(Y)
    X = np.squeeze(ARaw_B)

    # Compute K calibration gains using linear regression closed form
    K_accel = (np.linalg.inv(X.T @ X) @ X.T @ Y).T


    print("K_accel", K_accel)
    ## Now solve for accel alpha and beta
    
    accel_alpha = np.zeros(3)
    accel_beta = np.zeros(3)

    for row in range(K_accel.shape[0]-1):
        k_i = K_accel[row][row]
        b_i = K_accel[row][-1]

        accel_alpha[row] = 3300 / (1023 * k_i)
        accel_beta[row] = - b_i / k_i

    return accel_alpha, accel_beta



def calibrate_gyroscope(gyro, rots, vicon, time):

    

    ### Calibrating the gyroscope

    T = time.shape[0] - 1
    
    ## Assemble the vector of rotation matrices

    R_WB = rots

    R_WB = np.transpose(R_WB, axes=(2, 0, 1))

    R_WB = np.pad(R_WB, ((0, 0), (0, 1), (0, 1)), mode='constant', constant_values=0)
    R_WB[:, 3, 3] = 1


    ## Preprocess omegaRaw_B

    omegaRaw_B = gyro.T

    omegaRaw_B = np.pad(omegaRaw_B, ((0, 0), (0, 1)), mode='constant', constant_values=1)

    omegaRaw_B = omegaRaw_B.reshape(-1, 1, 4)



    # Calculate omegaTrue_B

    omegaTrue_B = np.zeros((T-1, 1, 4))

    for i in range(T-2):

        R_i = R_WB[i]

        dt = time[i+1] - time[i]
        
        Rdot_i = (R_WB[i+1] - R_WB[i]) / dt

        # Rdot_i = (R_WB[i+1] @ R_WB[i].T) / dt

        omegaTrue_skew_i = (np.linalg.inv(R_i) @ Rdot_i)

        omegaTrue_W = np.hstack((unskew(omegaTrue_skew_i), [[1]]))

        omegaTrue_B[i] = omegaTrue_W @ R_WB[i+1]



    ## Trimming
    omegaTrue_B = omegaTrue_B[:-1, :]
    omegaRaw_B = omegaRaw_B[:-1, :]


    trim_range = (1000, 4000)
    omegaTrue_B = omegaTrue_B[trim_range[0]:trim_range[1], :]
    omegaRaw_B = omegaRaw_B[trim_range[0]:trim_range[1], :]



    X = np.squeeze(omegaRaw_B)
    Y = np.squeeze(omegaTrue_B)


    K_gyro = (np.linalg.inv(X.T @ X) @ X.T @ Y).T
    print("K_gyro\n", K_gyro)



    ## Now solve for accel alpha and beta
    
    gyro_alpha = np.zeros(3)
    gyro_beta = np.zeros(3)

    for row in range(K_gyro.shape[0]-1):
        k_i = K_gyro[row][row]
        b_i = K_gyro[row][-1]
        #
        # print("k_i", k_i)
        # print("b_i", b_i)

        pi = 3.1415

        gyro_alpha[row] = (3300 / (1023 * k_i))
        gyro_beta[row] = - b_i / k_i



    return gyro_alpha, gyro_beta


def calculate_roll_pitch(accel_B):
    roll = np.arctan2(accel_B[1], accel_B[2])
    pitch = np.arctan2(-accel_B[0], np.sqrt(accel_B[1]**2 + accel_B[2]**2))
    return roll, pitch


def generate_sigma_points(mean, covariance):

    n = 6

    sigma_points = np.zeros((2*n, 7))


    # Compute the square root of the covariance matrix (sort of like multidimensional standard deviation)
    sqrt_cov = scipy.linalg.sqrtm(covariance)


    # For each sigma point
    for i in range(2 * n):

        # Negative sign for second half of the 2n sigma points
        sign = 1 if i < n else -1

        # Scaling (not sure exactly what for)
        scale = np.sqrt(n)

        # Drop index i from range [0, 2n] to [0, n] (by taking remainder from i / n)
        scaled_idx = i % n

        # Extract column from sqrt_cov
        sqrt_cov_col = sqrt_cov[:, scaled_idx]

        ## Form the "multidimensional standard deviation" 6x1 vector
        sorta_std_dev = sign * scale * sqrt_cov_col

        # Get the 4x1 quaternion form of the stddev's 3x1 (axis-angle?) orientation
        orientation_std_dev_axis_angle = sorta_std_dev[:3]
        orientation_std_dev_quat = Quaternion()
        orientation_std_dev_quat.from_axis_angle(orientation_std_dev_axis_angle)

        # Rename variables for clarity
        orientation_mean_quat = mean[:4]
        angular_velocity_mean = mean[4:]
        angular_velocity_std_dev = sorta_std_dev[3:]

        ## Form the sigma point by "adding" the mean and this stddev together
        sigma_points[i] = np.array([orientation_mean_quat.__mul__(orientation_std_dev_quat),
                                    [angular_velocity_mean + angular_velocity_std_dev]])



    return sigma_points


def quaternion_weighted_mean(quats, weights):

    ## Calculate mean quaternion via equation 13 in http://www.acsu.buffalo.edu/%7Ejohnc/ave_quat07.pdf

    # def quaternion_weighted_sum(Q: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    # """ Quaternion weighted mean as per http://www.acsu.buffalo.edu/~johnc/ave_quat07.pdf.
    # Args:
    #     Q: a (n, 4) array of scalar-first quaternions
    #     weights: a (n,) array of weights. They need not sum to 1.
    # Return:
    #     A (4,) array for the resulting quaternion.
    # """
    # A = sum(w * (np.outer(q, q)) for q, w in zip(Q, weights))
    # # Get the eigenvector corresponding to largest eigenvalue.
    # return np.linalg.eigh(A)[1][:, -1]



    M = np.zeros((4, 4))

    for q, w in zip(quats, weights):

        # add weighted outer product of q w/ itself
        M += w * q @ q.T


    # CHECK: that this gets the evec for the LARGEST eval
    mean_quaternion = np.linalg.eig(M)[1][:, -1]
    
    return mean_quaternion





def get_distribution_from_quaternions(quats, prev_quat):

    # TODO: weights?


    ## Initialize
    mean = prev_quat
    n = 6
    Errors = np.zeros((3, 2 * n))


    ## Iteratively:
    for i in range(1000):

        ## Compute errors ei for ea/ sigma point, update E
        for quat_idx in range(quats.shape[0]):

            quat = quats[i]

            ei = quat @ mean.inv()
            Errors[:, quat_idx] = ei

        
        ## Compute standard mean of errors, use quat form as new mean
        mean_error = np.mean(Errors, axis=1)
        mean = Quaternion()
        mean.from_axis_angle(mean_error)


        ## Check if error is small enough to terminate
        threshold = 1e-4
        if (np.linalg.norm(mean_error) < threshold):

            ## Compute covariance
            covariance = np.zeros((3, 3))
            w = 1 / (2 * n)
            for e in Errors:
                covariance += e @ e.T

            ## Stop iterating
            break

    return mean, covariance


def predict_state_distribution(sigma_points, x_kgk):
    
    n = 6
    weights = np.full((2 * n), 1/(2 * n))


    mean_omega = np.zeros(3)
    cov_omega = np.zeros((3, 3))
    ## Calculate omega's distribution via weighted mean
    for i in range(2 * n):
        w = weights[i]
        x = sigma_points[i]

        mean_omega += w * f(x)[4:]

        cov_omega += w * (f(x)[4:] - mean_omega) @ (f(x)[4:] - mean_omega).T


    ## Calculate quaternion
    sigma_quats = sigma_points[:, :4]
    prev_quat = x_kgk[:4]
    mean_quat, cov_quat = get_distribution_from_quaternions(sigma_quats, prev_quat)
    

    

        






    mean = np.vstack((mean_quat, 
                      mean_omega))

    covariance = np.block([[cov_omega, 0],
                           [0, cov_quat]])

    return mean, covariance

# Dynamics (process model)
def f(x_k):



    return x_kp1



def g(x_kgk):


    omega_k = x_kgk[4:]

    q_k = Quaternion(scalar=x_kgk[0], vec=x_kgk[1:4])

    g = Quaternion(scalar=0, vec=[0, 0, -9.81])

    b = Quaternion(scalar=0, vec=[0, 1, 0])

    g_prime = q_k @ g @ q_k.inv()
    b_prime = q_k.inv() @ b @ q_k

    # set placeholder values for measurement noise
    v_rot = np.full(3, 0.01)
    v_acc = np.full(3, 0.01)
    v_mag = np.full(3, 0.01)

    y_rot = omega_k + v_rot
    y_acc = g_prime + v_acc
    y_mag = b_prime + v_mag


    y_kp1gk = np.vstack((y_rot, 
                         y_acc, 
                         y_mag))
    

    return y_kp1gk




def estimate_rot(data_num=1):
    #load data
    imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
    vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
    accel = imu['vals'][0:3,:].astype(float)
    gyro = imu['vals'][3:6,:].astype(float)
    rots = vicon["rots"].astype(float)
    T = np.shape(imu['ts'])[1]

    ## Data pre-processing

    time = imu['ts'].astype(float)
    time = np.squeeze(time)
    time = time - time[0]

    # Set print options
    np.set_printoptions(precision=4, suppress=True)

    # Trim everything up to time T
    T = min(T, accel.shape[1], gyro.shape[1], rots.shape[2])
    accel = accel[:, :T-1]
    gyro = gyro[:, :T-1]
    rots = rots[:, :, :T-1]
    time = time[:T-1]

    # Reorder the gyro readings (cuz they come in as Z, X, Y for some reason) 
    gyro = gyro[[1, 2, 0], :]

    # Negate x and y aces (cuz this imu is special)
    # gyro[0, :] = -gyro[0, :]
    # gyro[1, :] = -gyro[1, :]
    accel[0, :] = -accel[0, :]
    accel[1, :] = -accel[1, :]



    R_WB_vicon = rots
    R_WB_vicon = np.transpose(R_WB_vicon, axes=(2, 0, 1))

    print("R_WB_vicon.shape", R_WB_vicon.shape)


    Q_WB_vicon = np.zeros((T-1), dtype=Quaternion)
    for t in range(T-1):
        q = Quaternion()
        q.from_rotm(R_WB_vicon[t])
        Q_WB_vicon[t] = q

    print("Q_WB_vicon.shape", Q_WB_vicon.shape)
    print("Q_WB_vicon[0]", Q_WB_vicon[0])


    rpy_vicon = np.zeros((T-1, 3), dtype=Quaternion)

    for t in range(T-1):
        q = Q_WB_vicon[t]
        rpy_vicon[t] = q.euler_angles()


    ## Calibrate the IMU

    accel_alpha, accel_beta = calibrate_accelerometer(accel, rots, vicon, time)
    print("accel_alpha: ", accel_alpha)
    print("accel_beta: ", accel_beta)

    gyro_alpha, gyro_beta = calibrate_gyroscope(gyro, rots, vicon, time)
    print("gyro_alpha: ", gyro_alpha)
    print("gyro_beta: ", gyro_beta)


    ## Get calibrated data

    print("accel[:, 200]", accel[:, 200])


    accel_alpha = np.full(3, 35)
    accel_beta = np.array([-500, -500, 500])

    
    A_B = np.copy(accel) # Creates a copy rather than a reference
    for i in range(3):
        A_B[i] = (A_B[i] - accel_beta[i]) * (3300 / (1023 * accel_alpha[i]))

    omega_B = np.copy(gyro)
    for i in range(3):
        omega_B[i] = (omega_B[i] - gyro_beta[i]) * (3300 / (1023 * gyro_alpha[i]))
 
    print("accel[:, 200]", accel[:, 200])
    print("A_B[:, 200]", A_B[:, 200])



    ## Estimate roll and pitch
    rpy = np.zeros((T-1, 3))
    for t in range(T-1):
        roll, pitch = calculate_roll_pitch(A_B[:, t])
        rpy[t, 0] = roll
        rpy[t, 1] = pitch
        rpy[t, 2] = 0 # can't determine yaw w/ only gravity


    ## Initialize mean and covariance
    
    mean = np.zeros(7)
    covariance = np.full((6, 6), 0.01)


    ## For each time step, k
    for k in range(T-2):


        ### Predict next state


        ## Add process noise R to covariance prior to generating sigma points
        covariance += R

        ## Generate sigma points
        sigma_points = generate_sigma_points(mean, covariance)

        ## Transform sigma points w/ nonlinear dynamics
        propagated_sigma_points = np.zeros_like(sigma_points)

        for i in range(sigma_points.shape[0]):

            x_kgk = sigma_points[i]

            x_kp1gk = f(x_kgk)

            propagated_sigma_points[i] = x_kp1gk



        ## Construct new state estimate based on transformed sigma points
        mu_kp1gk, sigma_kp1gk = predict_state_distribution(propagated_sigma_points)




        ### Update prediction w/ observation


        ## Predict next observation w/ sigma points

        # Obtain calibrated accel and gyro readings
        accel = A_B[k+1]
        omega = omega_B[k+1]
        n = 6
        observations = np.zeros((7, 2 * n))

        for obs_idx in propagated_sigma_points:
            mu_kp1gk = propagated_sigma_points[obs_idx]
            observations[obs_idx] = g(mu_kp1gk)

        mean_observation = np.mean(observations, axis=1)

        sigma_yy = Q
        sigma_xy = np.zeros_like(Q)
        weight = 1 / (2 * n)
        for obs_idx in observations:
            obs = observations[obs_idx]
            mean_obs = mean_observation
            sigma_yy += (obs - mean_obs) @ (obs - mean_obs).T

            x_i = propagated_sigma_points[obs_idx]
            sigma_xy += (x_i - mu_kp1gk) @ (obs - mean_obs).T

        
        y_hat = mean_observation


        roll, pitch = calculate_roll_pitch(A_B[k+1])
        yaw = 0
        measured_rpy = np.array([[roll],
                                 [pitch],
                                 [yaw]])
        measured_omega = omega_B[k+1]
        y_imu = np.vstack((measured_rpy,
                           measured_omega))
        innovation = y_imu - y_hat

        K = sigma_xy @ np.linalg.inv(sigma_yy)

        mu_kp1gkp1 = mu_kp1gk + K @ innovation


        sigma_kp1gkp1 = sigma_kp1gk - K @ sigma_yy @ K.T


        




        ## Compute Kalman Gain

        ## Update state estimate (mu and sigma)







    ## Plot to compare with Vicon
        
    plt.figure()
    # RPY VICON
    plt.plot(time, rpy_vicon[:, 0], label="roll_vicon", linestyle='solid', color='r')
    plt.plot(time, rpy_vicon[:, 1], label="pitch_vicon", linestyle='solid', color='g')
    plt.plot(time, rpy_vicon[:, 2], label="yaw_vicon", linestyle='solid', color='b')
    # RPY ESTIMATED
    plt.plot(time, rpy[:, 0], label="roll", linestyle='dotted', color='r')
    plt.plot(time, rpy[:, 1], label="pitch", linestyle='dotted', color='g')
    plt.plot(time, rpy[:, 2], label="yaw", linestyle='dotted', color='b')

    plt.title("RPY estimated vs vicon")
    plt.legend()
    plt.show()











    # roll, pitch, yaw are numpy arrays of length T
    # return roll,pitch,yaw






estimate_rot(1)
