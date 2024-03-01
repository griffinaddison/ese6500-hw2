import numpy as np
from scipy import io
import scipy
from quaternion import Quaternion   # numpy-quaternio
import math




import matplotlib.pyplot as plt



import time as TIME

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

    # print("Y.shape", Y.shape)
    # print("X.shape", X.shape)
    # print("(X.T @ X).shape", (X.T @ X).shape)

    # Compute K calibration gains using linear regression closed form
    K_accel = (np.linalg.inv(X.T @ X) @ X.T @ Y).T


    # print("K_accel", K_accel)
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

    # print("X.shape", X.shape)
    # print("Y.shape", Y.shape)
    # print("(X.T @ X).shape", (X.T @ X).shape)


    K_gyro = (np.linalg.inv(X.T @ X) @ X.T @ Y).T
    # print("K_gyro\n", K_gyro)



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

    sigma_points = np.zeros((2*n, 7, 1))


    # Compute the square root of the covariance matrix (sort of like multidimensional standard deviation)
#
    # print("covariance", covariance)
    sqrt_cov = scipy.linalg.sqrtm(covariance)
    # print("sqrt_cov\n", sqrt_cov)

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
        orientation_mean_quat = Quaternion(scalar=mean[0], vec=mean[1:4])
        angular_velocity_mean = mean[4:]
        angular_velocity_std_dev = sorta_std_dev[3:]

        # print("orientation_mean_quat", orientation_mean_quat)
        # print("orientation_std_dev_quat", orientation_std_dev_quat)
        sigma_point_quat = orientation_mean_quat.__mul__(orientation_std_dev_quat)
        sigma_point_quat.normalize()
        sigma_point_quat = sigma_point_quat.q.reshape(4, 1)

        sigma_point_omega = (angular_velocity_mean + angular_velocity_std_dev).reshape(3, 1)
    
        # print("angular_velocity_mean\n", angular_velocity_mean)
        # print("angular_velocity_std_dev\n", angular_velocity_std_dev)

        # print("sigma_point_quat", sigma_point_quat)
        # print("sigma_point_omega", sigma_point_omega)

        sigma_point = np.vstack((sigma_point_quat, sigma_point_omega))
        # print("sigma_point", sigma_point)

        ## Form the sigma point by "adding" the mean and this stddev together

        # print("generated sigma point: \n", sigma_point )
        sigma_points[i] = sigma_point

    # print("sigma_points.shape", sigma_points.shape)
    # print("\nsigma_points\n", sigma_points)
    return sigma_points


def quaternion_weighted_mean(quats):

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



    # M = np.zeros((4, 4))

    w = 1/12
    for q in quats:

        # add weighted outer product of q w/ itself

        # q = Quaternion( scalar=q[0], vec=q[1:4])


        M += w * q @ q.T


    # CHECK: that this gets the evec for the LARGEST eval
    mean_quaternion = np.linalg.eig(M)[1][:, -1]
    
    return Quaternion(scalar=mean_quaternion[0], vec=mean_quaternion[1:4])





def get_distribution_from_quaternions(quats, prev_quat):
    ## Initialize
    mean = Quaternion(prev_quat[0], prev_quat[1:4])
    covariance = np.eye(3) * 0.0000
    n = 6
    Errors = np.zeros((2 * n, 3))
    ## Iteratively:
    for i in range(10000):
        for quat_idx in range(2 * n):
            quat = Quaternion(scalar=quats[quat_idx, 0, 0], 
                              vec=quats[quat_idx, 1:4, 0])
            ei_quat = quat * mean.inv()
            ei_quat.normalize()
            ei = ei_quat.axis_angle()

            # Errors[quat_idx, :] = ei 
            adjustment = (np.mod(np.linalg.norm(ei) + np.pi, 2 * np.pi) - np.pi) / np.linalg.norm(ei)
            Errors[quat_idx, :] = ei * adjustment 
        ei_mean = np.mean(Errors, axis=0)
        ei_mean_quat = Quaternion()
        ei_mean_quat.from_axis_angle(ei_mean)
        mean = (ei_mean_quat * mean)
        # mean.normalize()
        ## Check if error is small enough to terminate
        threshold = 1e-4
        if (np.linalg.norm(ei_mean) < threshold):

            # print("norm error:", np.linalg.norm(ei_mean))
            covariance = np.eye(3) * 0.000
            w = 1 / (2 * n)
            for e in Errors:
                covariance += w * e @ e.T
            return mean, covariance
        # Errors = np.zeros((2 * n, 3))
    w = 1 / (2 * n)
    covariance = np.zeros((3, 3))
    for e in Errors:
        covariance += w * e @ e.T
    return mean, covariance


def predict_state_distribution(sigma_points, curr_state, dt):
    
    n = 6
    weights = np.full((2 * n), 1/(2 * n))


    mean_omega = np.zeros((3, 1))
    cov_omega = np.zeros((3, 3))
    ## Calculate omega's distribution via weighted mean
    for i in range(2 * n):
        w = weights[i]
        x = sigma_points[i]

        mean_omega += w * f(x, dt)[4:]

        ## TODO: maybe use np.cov cuz maybe quicker
        cov_omega += w * (f(x, dt)[4:] - mean_omega) @ (f(x, dt)[4:] - mean_omega).T


    ## Calculate quaternion
    sigma_quats = sigma_points[:, :4]
    prev_quat = curr_state[:4]
    # time_start_get_distribution_from_quaternions = TIME.perf_counter()
    mean_quat, cov_quat = get_distribution_from_quaternions(sigma_quats, prev_quat)

    # mean_quat = quaternion_weighted_mean(sigma_quats)
    
    # time_end_get_distribution_from_quaternions = TIME.perf_counter()
    # print("\n time to get distribution from quaternions: ", time_end_get_distribution_from_quaternions - time_start_get_distribution_from_quaternions)

    # print("HIIIIIIIIIII")

    # print("mean_quat.q", mean_quat.q)
    # print("mean_omega", mean_omega)
    mean = np.vstack((mean_quat.q.reshape(4, 1),
                      mean_omega))

    covariance = np.block([[cov_quat, np.zeros_like(cov_quat)],
                           [np.zeros_like(cov_omega), cov_omega]])

    return mean, covariance

# Dynamics (process model)
def f(x_k, dt):


    q_k = Quaternion(x_k[0, 0], x_k[1:4, 0])
    omega_k = x_k[4:, 0]

    # print("x_k\n", x_k)
    # print("dt", dt)
    # print("omega_k\n", omega_k)
    # print("np.linalg.norm(omega_k)\n", np.linalg.norm(omega_k))
    #
    # q_delta = Quaternion(scalar=cos(alpha_delta), vec=0.5 * omega_k)


    # alpha_delta = np.linalg.norm(omega_k) * dt
    # e_delta = omega_k / np.linalg.norm(omega_k) if not np.linalg.norm(omega_k) == 0 else np.array([1, 0, 0])
    # q_delta = Quaternion(scalar= np.cos(alpha_delta/2), 
    #                      vec= e_delta * np.sin(alpha_delta/2))
    #
    q_delta = Quaternion()
    q_delta.from_axis_angle(omega_k * dt)

    q_kp1 = q_k * q_delta
    # q_kp1.normalize()
    q_kp1 = q_kp1.q.reshape(4, 1)

    omega_kp1 = omega_k.reshape(3, 1)

    x_kp1 = np.vstack((q_kp1,
                          omega_kp1))

    return x_kp1



def g(state):


    q_k = Quaternion(scalar=state[0, 0], 
                     vec=state[1:4, 0])

    # TODO: maybe change sign of 9.81
    g = Quaternion(scalar=0, 
                   vec=[0, 0, 9.81])

    g_prime = q_k.inv() * g * q_k
    g_prime.normalize()

    y_rot = state[4:]
    y_acc = g_prime.vec().reshape(-1, 1)



    # print("y_rot.shape", y_rot.shape)
    # print("y_acc.shape", y_acc.shape)
    # y_kp1gk = np.vstack((y_rot, 
    #                      y_acc))
    y_kp1gk = np.vstack((y_acc,
                         y_rot))  

    return y_kp1gk






























def get_calibrated_imu_data(data_num):

    #load data
    imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
    time = imu['ts'].astype(float)
    time = np.squeeze(time)
    time = time - time[0]
    T = time.shape[0]

    accel = imu['vals'][0:3,:].astype(float)

    accel[0] = -accel[0]
    accel[1] = -accel[1]
    accel = accel.T

    gyro = imu['vals'][3:6,:].astype(float)
    gyro = gyro.T
    gyro = gyro[:, [1, 2, 0]]


    accel = accel[:T]
    gyro = gyro[:T]
    time = time[:T]

  
    

    accel_alpha = np.full(3, 350.0)
    accel_beta = np.array([-510.0, -500.0, 500.0])

    # gyro_alpha = np.full(3, 200)
    gyro_alpha = np.full(3, 3.33)
    gyro_beta = np.array([375, 375, 375])

    accel = (accel - accel_beta) * (3300 / (1023 * accel_alpha))
    gyro = (gyro - gyro_beta) * (3300.0 / (1023.0 * gyro_alpha)) * (np.pi/180)

    
    imu_observations = np.zeros((T, 6, 1))
    
    # print("accel[0]", accel[0])
    # print("accel.shape\n", accel.shape)
    # print("gyro\n", gyro.shape)
    # print("accel\n", accel)
    # print("gyro\n", gyro)

    imu_observations = np.hstack((accel.reshape(-1, 3, 1),
                                  gyro.reshape(-1, 3, 1)))

    # print("imu_observations[0]", imu_observations[0])
    # print("imu_observations.shape", imu_observations.shape)
    # print("imu_observations", imu_observations)

    # for k in range(T-1):
    #     roll_cal, pitch_cal = calculate_roll_pitch(accel[k, :])
    #     yaw_cal = 0
    #
    #     print("accel[k, :]", accel[k, :])
    #     print("roll: ", roll_cal)
    #     print("pitch: ", pitch_cal)
    #     imu_rpy = np.array([[roll_cal],
    #                         [pitch_cal],
    #                         [yaw_cal]])
    #     imu_gyro = gyro[k, :].reshape(3, 1)
    #     imu_observations[i] = np.vstack((imu_rpy,
    #                                      imu_gyro))
    #
    return accel, gyro, time, imu_observations


def initialize_variables():


    quat = Quaternion()
    quat.from_axis_angle(np.array([0, 0, 0]))
    mean = np.hstack((quat.q, [0, 0, 0]))

    print("mean", mean)

    covariance = np.eye(6) * 0.1

    R = np.eye(6) * 0.001
    Q = np.eye(6) * 0.001

    n = 6
    
    return mean, covariance, R, Q, n


    



def estimate_rot(data_num=1, time_steps=999999):

    mean, covariance, R, Q, n = initialize_variables()

    accel, gyro, time, imu_observations = get_calibrated_imu_data(data_num)

    T = min(time.shape[0], time_steps)

    tic = TIME.perf_counter()
    setup_time = TIME.perf_counter() - tic


    roll = np.zeros((T))
    pitch = np.zeros((T))
    yaw = np.zeros((T))


    ## For each time step, 
    for k in range(T-1):
    # for k in range(2):
        
        if k % 500 == 0:
            step_start_time = TIME.perf_counter()
            print("\n time step k = ", k)


        dt = max(time[k+1] - time[k], 0.0001)

        sigma_points = generate_sigma_points(mean, covariance + R * dt) 
        propagated_sigma_points = np.zeros_like(sigma_points)
        for i in range(sigma_points.shape[0]):
            propagated_sigma_points[i] = f(sigma_points[i], dt)
      
        predicted_mean, predicted_covariance = predict_state_distribution(propagated_sigma_points, mean, dt)
       
        predicted_observations = np.zeros((2 * n, 6, 1))
        for i in range(sigma_points.shape[0]):
            # predicted_observations[i] = g(f(sigma_points[i], dt))
            predicted_observations[i] = g(propagated_sigma_points[i])
        
        mean_predicted_observation = np.mean(predicted_observations, axis=0)
        sigma_yy = Q
        sigma_xy = np.zeros_like(sigma_yy)
        weight = 1 / (2 * n)
        for i in range(predicted_observations.shape[0]):

            sigma_yy += weight * (predicted_observations[i] - mean_predicted_observation)\
                                    @ ((predicted_observations[i] - mean_predicted_observation).T)
            quat_i = Quaternion(propagated_sigma_points[i, 0, 0],
                                propagated_sigma_points[i, 1:4, 0])
            quat_mean = Quaternion(predicted_mean[0, 0], 
                                   predicted_mean[1:4, 0])
            quat_error = quat_i * quat_mean.inv()
            state_error = np.vstack((quat_error.axis_angle().reshape(3, 1),
                                     propagated_sigma_points[i, 4:] - predicted_mean[4:]))
            sigma_xy += weight * (state_error)\
                            @ ((predicted_observations[i] - mean_predicted_observation).T)



        ## Compute Kalman Gain
        # Obtain calibrated accel and gyro readings
        # roll_cal, pitch_cal = calculate_roll_pitch(accel[k+1, :])
        # yaw_cal = 0
        # imu_rpy = np.array([[roll_cal],
        #                     [pitch_cal],
        #                     [yaw_cal]])
        # imu_gyro = gyro[k+1, :].reshape(3, 1)
        # imu_observations = np.vstack((imu_rpy,
        #                              imu_gyro))

        # print("imu_observations[k]", imu_observations[k])
        # print("mean_predicted_observation", mean_predicted_observation)

        innovation = imu_observations[k+1] - mean_predicted_observation
        K = sigma_xy @ np.linalg.inv(sigma_yy)
        ## Update state estimate (mu and sigma)
        Kinnovation = K @ innovation
        Kinnovation_quat = Quaternion()
        Kinnovation_quat.from_axis_angle(Kinnovation[:3].squeeze())
        predicted_mean_quat = Quaternion(scalar= predicted_mean[0, 0],
                                         vec= predicted_mean[1:4, 0])
        updated_mean_quat = Kinnovation_quat * predicted_mean_quat
        updated_mean_omega = predicted_mean[4:] + Kinnovation[3:]
        updated_mean = np.vstack((updated_mean_quat.q.reshape(4, 1),
                                  updated_mean_omega))
        updated_orientation = Quaternion(updated_mean[0, 0], updated_mean[1:4, 0])
        updated_orientation.normalize()
        updated_covariance = predicted_covariance - K @ sigma_yy @ K.T
        # small_positive_value = 0.001
        # np.fill_diagonal(updated_covariance,
        #                  np.maximum(np.diagonal(updated_covariance), small_positive_value))
        #
        mean = updated_mean.squeeze()
        covariance = updated_covariance

        rpy_ukf = updated_orientation.euler_angles()

        roll[k+1] = rpy_ukf[0]
        pitch[k+1] = rpy_ukf[1]
        yaw[k+1] = rpy_ukf[2]


        if k % 500 == 0:
            step_end_time = TIME.perf_counter() - step_start_time
            print("\n step time: ", step_end_time)


    ## Sanitize outputs
    # roll = roll[np.isfinite(roll)]
    # pitch = pitch[np.isfinite(pitch)]
    # yaw = yaw[np.isfinite(yaw)]
    #
    # roll = roll[~np.isnan(roll)]
    # pitch = pitch[~np.isnan(pitch)]
    # yaw = yaw[~np.isnan(yaw)]

    return roll,pitch,yaw






def get_vicon_data(data_num=1): 

    
    vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
    rots = vicon["rots"].astype(float)


    vicon_rotation_matrices = vicon["rots"].astype(float).transpose((2, 0, 1))

    vicon_orientations = np.zeros(vicon_rotation_matrices.shape[0], dtype=Quaternion)
    for i in range(vicon_rotation_matrices.shape[0]):
        q = Quaternion()
        q.from_rotm(vicon_rotation_matrices[i])
        vicon_orientations[i] = q

    vicon_orientations_rpy = np.zeros((vicon_orientations.shape[0], 3))

    for t in range(vicon_orientations_rpy.shape[0]):
        q = vicon_orientations[t]
        vicon_orientations_rpy[t] = q.euler_angles()


    
    imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
    ## Data pre-processing
    time = imu['ts'].astype(float)
    time = np.squeeze(time)
    time = time - time[0]



    return vicon_orientations_rpy, vicon_orientations, time




def plotRpy(data_num=1, time_steps=999999):



    ukf_orientations_rpy = estimate_rot(data_num, time_steps)
    _, _, _, imu_observations = get_calibrated_imu_data(data_num)

    vicon_orientations_rpy, vicon_orientations, time = get_vicon_data(data_num)

    T = min(ukf_orientations_rpy[0].shape[0], vicon_orientations_rpy.shape[0], time_steps)

    imu_rpy = np.zeros((T, 3))
    for i in range(T-1):
        r, p = calculate_roll_pitch(imu_observations[i, :3])
        imu_rpy[i, 0] = r
        imu_rpy[i, 1] = p
        imu_rpy[i, 2] = 0
 

    plt.figure()
    plt.subplot(3, 1, 1)
    # ROLL
    plt.plot(time[:T], vicon_orientations_rpy[:T, 0], 
             label="roll_vicon", linestyle='solid', color='r')
    plt.plot(time[:T], ukf_orientations_rpy[0][:T], 
             label="roll_ukf", linestyle='solid', color='r', alpha=0.4)
    plt.plot(time[:T], imu_rpy[:T, 0],
             label="roll_imu", linestyle='dotted', color='r', alpha=0.4)
    plt.legend()
    plt.subplot(3, 1, 2)
    # PITCH
    plt.plot(time[:T], vicon_orientations_rpy[:T, 1], 
             label="pitch_vicon", linestyle='solid', color='g')
    plt.plot(time[:T], ukf_orientations_rpy[1][:T], 
             label="pitch_ufk", linestyle='solid', color='g', alpha=0.4)
    # YAW
    plt.subplot(3, 1, 3)
    plt.plot(time[:T], vicon_orientations_rpy[:T, 2], 
             label="yaw_vicon", linestyle='solid', color='b')
    plt.plot(time[:T], ukf_orientations_rpy[2][:T], 
             label="yaw_ufk", linestyle='solid', color='b', alpha=0.4)

    
    plt.title("RPY vicon vs UKF")
    plt.legend()
    plt.show()




plotRpy(1)



