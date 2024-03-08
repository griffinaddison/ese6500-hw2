import numpy as np
from scipy import io
import scipy
from quaternion import Quaternion   # numpy-quaternio
import math

import matplotlib.pyplot as plt


#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an unscented kalman filter


''' Implemented below is an Unscented Kalman Filter that aims to estimate the orientation of a falling IMU in 3D space. '''

''' Implemented by Griffin Addison in early March, 2024. '''

''' The function 'estimate_rot()' is where main, high-level UKF loop occurs. '''


def unskew(m):
    return np.array([[(m[2][1] - m[1][2])/2, (m[0][2] - m[2][0])/2, (m[1][0] - m[0][1])/2]])

def get_accelerometer_calibration(data_num=1):
   

    #load data
    imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
    time = imu['ts'].astype(float)
    time = np.squeeze(time)
    time = time - time[0]
    # T = time.shape[0]

    vicon_rotation_matrices, _, _, _, time_vicon = get_vicon_data(data_num)
    accel = imu['vals'][0:3,:].astype(float)
    accel[0] = -accel[0]
    accel[1] = -accel[1]
    accel = accel.T

    gyro = imu['vals'][3:6,:].astype(float)
    gyro = gyro.T
    gyro = gyro[:, [1, 2, 0]]


    T = min(accel.shape[0], gyro.shape[0], vicon_rotation_matrices.shape[0], time.shape[0])

    gyro = gyro[:T]

    time = time[:T]
    accel = accel[:T]
    vicon_rotation_matrices = vicon_rotation_matrices[:T]


    ARaw_B = accel

    ARaw_B = np.pad(ARaw_B, ((0, 0), (0, 1)), mode='constant', constant_values=1)

    ARaw_B = ARaw_B.reshape((-1, 1, 4))

    R_WB = np.copy(vicon_rotation_matrices)

    R_WB = np.pad(R_WB, ((0, 0), (0, 1), (0, 1)), mode='constant', constant_values=0)
    R_WB[:, 3, 3] = 1

    R_WB_transpose = np.transpose(R_WB, axes=(0, 2, 1))


    # Assemble the vector of ground truth a_W (just gravity) A_W
    A_W = np.zeros((T, 4, 1))
    A_W[:, 2, 0] = 9.81
    A_W[:, 3, 0] = 1

    # Create Y and X based upon our problem's variables
    Y = np.matmul(R_WB_transpose, A_W)
    Y = np.squeeze(Y)
    X = np.squeeze(ARaw_B)

    # Compute K calibration gains using linear regression closed form
    K_accel = (np.linalg.inv(X.T @ X) @ X.T @ Y).T

    
    accel_alpha = np.zeros(3)
    accel_beta = np.zeros(3)

    for row in range(K_accel.shape[0]-1):
        k_i = K_accel[row][row]
        b_i = K_accel[row][-1]

        accel_alpha[row] = 3300 / (1023 * k_i)
        accel_beta[row] = - b_i / k_i


    return accel_alpha, accel_beta


def get_gyroscope_calibration(data_num=1):

        #load data
    imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
    time = imu['ts'].astype(float)
    time = np.squeeze(time)
    time = time - time[0]

    vicon_rotation_matrices, _, _, _, time_vicon = get_vicon_data(data_num)
    accel = imu['vals'][0:3,:].astype(float)
    accel[0] = -accel[0]
    accel[1] = -accel[1]
    accel = accel.T

    gyro = imu['vals'][3:6,:].astype(float)
    gyro = gyro.T
    gyro = gyro[:, [1, 2, 0]]


    T = min(accel.shape[0], gyro.shape[0], vicon_rotation_matrices.shape[0], time.shape[0])


    gyro = gyro[:T]
    vicon_rotation_matrices = vicon_rotation_matrices[:T]
    time = time[:T]
    accel = accel[:T]


    ## Formatting the data for linear regression step

    
    R_WB = np.copy(vicon_rotation_matrices)

    R_WB = np.pad(R_WB, ((0, 0), (0, 1), (0, 1)), mode='constant', constant_values=0)
    R_WB[:, 3, 3] = 1



    omegaRaw_B = np.copy(gyro)

    omegaRaw_B = np.pad(omegaRaw_B, ((0, 0), (0, 1)), mode='constant', constant_values=1)

    omegaRaw_B = omegaRaw_B.reshape(-1, 1, 4)


    omegaTrue_B = np.zeros((T, 1, 4))
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
    
    gyro_alpha = np.zeros(3)
    gyro_beta = np.zeros(3)

    for row in range(K_gyro.shape[0]-1):
        k_i = K_gyro[row][row]
        b_i = K_gyro[row][-1]
       
        gyro_alpha[row] = (3300 / (1023 * k_i))# * (np.pi/180)
        gyro_beta[row] = - b_i / k_i
    return gyro_alpha, gyro_beta

def get_calibrated_imu_data(data_num, recalibrate=False):

    #load data
    imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
    time = imu['ts'].astype(float)
    time = np.squeeze(time)
    time = time - time[0]
    T = time.shape[0]
    time = time[:T]

    accel = imu['vals'][0:3,:].astype(float)
    accel[0] = -accel[0]
    accel[1] = -accel[1]
    accel = accel.T
    accel = accel[:T]

    gyro = imu['vals'][3:6,:].astype(float)
    gyro = gyro.T
    gyro = gyro[:, [1, 2, 0]]
    gyro = gyro[:T]


    if recalibrate:

        accel_alphas, accel_betas = np.zeros((3, 3)), np.zeros((3, 3))
        gyro_alphas, gyro_betas = np.zeros((3, 3)), np.zeros((3, 3))
        for data_num in range(1, 4):
            accel_alphas[data_num-1], accel_betas[data_num-1] = get_accelerometer_calibration(data_num)
            gyro_alphas[data_num-1], gyro_betas[data_num-1] = get_gyroscope_calibration(data_num)

            print("\n Calibrated accelerometer alphas: \n", accel_alphas)
            print("\n Calibrated accelerometer betas: \n", accel_betas)
            print("\n Calibrated gyroscope alphas: \n", gyro_alphas)
            print("\n Calibrated gyroscope betas: ", gyro_betas)


    else:

        # Hardcoded values
        # accel_alpha = np.full(3, 35.0)
        accel_beta = np.array([-510.0, -500.0, 500.0])
        #
        # gyro_alpha = np.full(3, 200)
        # gyro_beta = np.array([347, 347, 347])

        # Hardcoded values
        accel_alpha = np.array([35.5, 35.0, 36.5])
        accel_beta = np.array([-510.0, -502.0, 489.0])

        gyro_alpha = np.array([259.0, 284.0, 274])
        gyro_beta = np.array([339.5, 358.0, 350.0])

    accel = (accel - accel_beta) * (3300 / (1023 * accel_alpha))
    gyro = (gyro - gyro_beta) * (3300.0 / (1023.0 * gyro_alpha)) * (np.pi/180)

    
    imu_observations = np.zeros((T, 6, 1))
    
    imu_observations = np.hstack((accel.reshape(-1, 3, 1),
                                  gyro.reshape(-1, 3, 1)))

    return accel, gyro, time, imu_observations


def vec2quat(vec):
    if len(vec) < 4:
        raise ValueError("Cannot convert vec of len ", len(vec), " to quat.")
    else:
        vec = vec.squeeze()
        q = Quaternion(scalar=vec[0], vec=vec[1:4])
        q.normalize()
        return q


def aa2quat(axis_angle, normalize=True):
    if len(axis_angle) < 3:
        raise ValueError("Cannot create quaternion from axis-angle vec w/ ",
              len(axis_angle), " elements.")
    else:
        axis_angle = axis_angle.squeeze()
        q = Quaternion()
        q.from_axis_angle(axis_angle)
        if normalize:
            q.normalize()
        return q




def calculate_roll_pitch(accel_B):
    roll = np.arctan2(accel_B[1], accel_B[2])
    pitch = np.arctan2(-accel_B[0], np.sqrt(accel_B[1]**2 + accel_B[2]**2))
    return roll, pitch


def generate_sigma_points(mu, cov):

    # TODO: maybe include mean as a sigma point
    n = 6
    Xi = np.zeros((2 * n, 7, 1))

    sqrt_cov = scipy.linalg.sqrtm(cov).real

    # For each sigma point
    for i in range(2 * n):

        # Drop index i from range [0, 2n] to [0, n] (by taking remainder from i / n)
        scaled_idx = i % n

        # Extract column from sqrt_cov
        sqrt_cov_col = sqrt_cov[:, scaled_idx]

        # Negative sign for second half of the 2n sigma points
        sign = 1 if i < n else -1

        # Form the "multidimensional standard deviation" 6x1 vector
        variation = sign * np.sqrt(n) * sqrt_cov_col

        variation_q = Quaternion(scalar=1, vec=variation[:3])
        
        # Create this sigma point by adding the variation to the mean
        xi_q = (vec2quat(mu[:4]) * variation_q).q.reshape(4, 1)
        xi_w = mu[-3:].reshape(3, 1) + variation[-3:].reshape(3, 1)

        Xi[i] = np.vstack((xi_q.reshape(-1, 1),
                            xi_w.reshape(-1, 1)))

    return Xi




def quat_average(quats, threshold=1e-5, quat_initial_guess=Quaternion()):
    ## Initialize
    mean = vec2quat(quat_initial_guess)
    covariance = np.eye(3) * 0.001
    n = 6
    Errors = np.zeros((2 * n, 3))
    ## Iteratively:
    for iter in range(5000):
        for i in range(2 * n):  
            quat = vec2quat(quats[i])
            ei_quat = quat * mean.inv()
            ei_quat.normalize()
            ei = ei_quat.axis_angle()

            if np.linalg.norm(ei) == 0:
                Errors[i:] = np.zeros(3)
            else:
                range_bounding = (np.mod(np.linalg.norm(ei) + np.pi, 2 * np.pi) - np.pi) / np.linalg.norm(ei)
                Errors[i, :] = ei * range_bounding

        ei_mean = np.mean(Errors, axis=0)
        ei_mean_quat = aa2quat(ei_mean, normalize=True)
        mean = (ei_mean_quat * mean)
        mean.normalize()
        ## Check if error is small enough to terminate
        if (np.linalg.norm(ei_mean) < threshold):
            covariance = np.eye(3) * 0.00
            w = 1 / (2 * n)
            # print("\nGD completed after ", iter, " iterations.")
            for e in Errors:
                covariance += w * e @ e.T
            return mean, covariance, Errors
    Errors = np.zeros((2 * n, 3))


def reconstruct_state_distribution(Xi, mu_kgk, dt):
    
    
    weight = 1 / len(Xi)

    ## Calculate omega's distribution via weighted mean
    mu_kp1gk_omega = np.zeros((3, 1))
    sigma_kp1gk_omega = np.zeros((3, 3))
    
    for i in range(len(Xi)):
        mu_kp1gk_omega += weight * Xi[i, -3:]
    for i in range(len(Xi)):
        sigma_kp1gk_omega += weight * (Xi[i, -3:] - mu_kp1gk_omega) \
                                    @ (Xi[i, -3:] - mu_kp1gk_omega).T

    ## Calculate quaternion
    Xi_quats = Xi[:, :4]
    

    mu_kp1gk_quat, sigma_kp1gk_quat, Errors = quat_average(Xi_quats, quat_initial_guess=mu_kgk[:4])

    mu_kp1gk = np.vstack((mu_kp1gk_quat.q.reshape(4, 1),
                      mu_kp1gk_omega))

    sigma_kp1gk = np.block([[sigma_kp1gk_quat, np.zeros((3, 3))],
                           [np.zeros((3, 3)), sigma_kp1gk_omega]])

    return mu_kp1gk, sigma_kp1gk, Errors

def initialize_variables():

    mean_quat = [1, 0, 0, 0]
    mean_omega = [0, 0, 0]
    mean = np.hstack((mean_quat, mean_omega))

    covariance = np.eye(6)
    R = np.eye(6) 
    Q = np.eye(6) * 5.0

    n = 6
    
    return mean, covariance, R, Q, n





def g(state):

    # Ready the variables
    q_k = Quaternion(scalar=state[0, 0], 
                     vec=state[1:4, 0])
    g = Quaternion(scalar=0, 
                   vec=[0, 0, 9.81])
    omega_k = state[4:]

    # Transform g into local(?) frame
    g_prime = q_k.inv() * g * q_k

    # Assign to outputs
    y_acc = g_prime.vec().reshape(-1, 1)
    y_rot = omega_k
    y_kp1gk = np.vstack((y_acc,
                         y_rot))  

    return y_kp1gk

def measurement_model(Xi):

    Yi = np.zeros((len(Xi), 6, 1))
    for i in range(Xi.shape[0]):
        # Calculate the measurement for this state 
        Yi[i] = g(Xi[i])

    return Yi


# Dynamics (process model)
def f(x_k, dt):

    # Readying variables
    q_k = vec2quat(x_k[:4])
    omega_k = x_k[4:, 0]
    q_delta = aa2quat(omega_k * dt)

    # predict next state w/ dynamics
    q_kp1 = q_k * q_delta
    omega_kp1 = omega_k

    x_kp1 = np.vstack((q_kp1.q.reshape(4, 1),
                          omega_kp1.reshape(3, 1)))
    return x_kp1

def process_model(Xi_kgk, dt):

    Xi_kp1gk = np.zeros_like(Xi_kgk)
    for i in range(Xi_kgk.shape[0]):
        # Transform the ith sigma point w/ process model (dynamics) 
        Xi_kp1gk[i] = f(Xi_kgk[i], dt)

    return Xi_kp1gk 







def estimate_rot(data_num=1, enable_plot=False, recalibrate=False, time_steps=999999):


    '''SETUP'''

    # Acquire IMU data
    accel, gyro, time, Y_imu = get_calibrated_imu_data(data_num, recalibrate=recalibrate)
    T = min(time.shape[0], time_steps)
    
    # Initialize tunables
    mu_kgk, sigma_kgk, R, Q, n = initialize_variables()

    # Initialize the state variables we will track
    ukf_orientations = np.zeros((T, 4))
    ukf_orientations[0, 0] = 1
    ukf_angular_velocities = np.zeros((T, 3))
    ukf_covariances = np.zeros((T, 6, 6))
    roll = np.zeros((T))
    pitch = np.zeros((T))
    yaw = np.zeros((T))


    '''FOR EACH TIMESTEP'''
    for k in range(T-1):
        
        if k % 500 == 0:
            print("\n Running UKF: ", round((k / T) * 100, 0), "% complete.\n")

        dt = time[k+1] - time[k]  


        '''PREDICT NEXT STATE DISTRIBUTION'''

        sigma_kgk += R * dt
        Xi = generate_sigma_points(mu_kgk, sigma_kgk)
        Xi_kp1gk = process_model(Xi, dt)
        mu_kp1gk, sigma_kp1gk, Errors = reconstruct_state_distribution(Xi_kp1gk, mu_kgk, dt)
      

        '''PREDICT NEXT OBSERVATION'''

        Yi = measurement_model(Xi_kp1gk)
        yi_bar = np.mean(Yi, axis=0)



        '''COMPARE PREDICTIONS WITH OBSERVATIONS TO GET COVARIANCES'''

        sigma_yy, sigma_xy = np.zeros((6, 6)), np.zeros((6, 6))
        weight = 1 / (2 * n)
        for i in range((2 * n)):
   
            sigma_yy += weight * np.outer(Yi[i] - yi_bar, Yi[i] - yi_bar)

            r_W = vec2quat(Xi_kp1gk[i, :4]) * vec2quat(mu_kp1gk[:4]).inv()
            Wi = np.vstack((r_W.vec().reshape(3, 1),
                            Xi_kp1gk[i, -3:] - mu_kp1gk[-3:]))


            sigma_xy += weight * np.outer(Wi, Yi[i] - yi_bar)

        sigma_yy += Q



        '''COMPUTE KALMAN GAIN TO UPDATE PREDICTION'''

        innovation = Y_imu[k+1] - yi_bar
        K = sigma_xy @ np.linalg.inv(sigma_yy)
        Kinnovation = K @ innovation

        q_kp1gkp1 = aa2quat(Kinnovation[:3]) * vec2quat(mu_kp1gk[:4]) 
        q_kp1gkp1.normalize()
        mu_kp1gkp1 = np.vstack((q_kp1gkp1.q.reshape(4, 1),
                                mu_kp1gk[-3:] + Kinnovation[-3:]))
        
        sigma_kp1gkp1 = sigma_kp1gk - K @ sigma_yy @ K.T

        

        '''CLEAN AND RECORD DATA'''

        ## Make sure quat sign doesn't flip
        prev_max_component_idx = np.argmax(np.abs(ukf_orientations[k]))
        prev_max_component = ukf_orientations[k, prev_max_component_idx]
        new_component = q_kp1gkp1.q[prev_max_component_idx]
        max_sign_flipped = np.sign(prev_max_component) != np.sign(new_component)
        if max_sign_flipped:
            q_kp1gkp1 = vec2quat(-1 * np.copy(q_kp1gkp1.q))
        else:
            q_kp1gkp1 = vec2quat(np.copy(q_kp1gkp1.q))


        # Ensure w component stays positive
        q_kp1gkp1 = vec2quat(np.sign(q_kp1gkp1.scalar()) * q_kp1gkp1.q)
        # q_kp1gkp1.normalize()
        ukf_orientations[k+1] = q_kp1gkp1.q

        ukf_angular_velocities[k+1] = mu_kp1gkp1[-3:].squeeze()

        ukf_covariances[k+1] = sigma_kp1gkp1


        roll[k+1], pitch[k+1], yaw[k+1] = q_kp1gkp1.euler_angles() 

        # For next loop:
        mu_kgk = np.copy(mu_kp1gkp1)
        sigma_kgk = np.copy(sigma_kp1gkp1)


    '''POST-LOOP HOUSEKEEPING'''

    ukf_orientations_rpy = roll, pitch, yaw


    if enable_plot:
        print("\n PLOTTING... \n")
        plot(ukf_orientations, ukf_angular_velocities, ukf_covariances, ukf_orientations_rpy, data_num)



    # Sanitize outputs
    roll = roll[np.isfinite(roll)]
    pitch = pitch[np.isfinite(pitch)]
    yaw = yaw[np.isfinite(yaw)]

    roll = roll[~np.isnan(roll)]
    pitch = pitch[~np.isnan(pitch)]
    yaw = yaw[~np.isnan(yaw)]

    return roll,pitch,yaw






def get_vicon_data(data_num=1): 

    
    vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
    rots = vicon["rots"].astype(float)


    vicon_rotation_matrices = vicon["rots"].astype(float).transpose((2, 0, 1))

    vicon_orientations = np.zeros((vicon_rotation_matrices.shape[0], 4))
    for i in range(vicon_rotation_matrices.shape[0]):
        q = Quaternion()
        q.from_rotm(vicon_rotation_matrices[i])
        vicon_orientations[i, :] = q.q

    vicon_orientations_rpy = np.zeros((vicon_orientations.shape[0], 3))

    for t in range(vicon_orientations_rpy.shape[0]):
        q = vec2quat(vicon_orientations[t])
        vicon_orientations_rpy[t] = q.euler_angles()


    
    imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
    ## Data pre-processing
    time = imu['ts'].astype(float)
    time = np.squeeze(time)
    time = time - time[0]

    T = vicon_rotation_matrices.shape[0]

    R_WB = vicon_rotation_matrices

    omegaTrue_B = np.zeros((T, 3))
    for i in range(T-2):
        
        # Ready the variables
        dt = time[i+1] - time[i]
        R_i = R_WB[i]
        Rdot_i = (R_WB[i+1] - R_WB[i]) / dt
    
        omegaTrue_skew_i = (np.linalg.inv(R_i) @ Rdot_i)

        omegaTrue_W = unskew(omegaTrue_skew_i)
        omegaTrue_B[i, :] = (omegaTrue_W @ R_WB[i+1]).squeeze()


    return vicon_rotation_matrices, omegaTrue_B, vicon_orientations_rpy, vicon_orientations, time



def plot(ukf_orientations, ukf_angular_velocities, ukf_covariances, ukf_orientations_rpy, data_num, time_steps=999999):



    '''READY THE DATA'''
    _, _, _, imu_observations = get_calibrated_imu_data(data_num)

    _, vicon_angular_velocities, vicon_orientations_rpy, vicon_orientations, time \
    = get_vicon_data(data_num)

    T = min(ukf_orientations_rpy[0].shape[0], vicon_orientations_rpy.shape[0], time_steps)

    imu_rpy = np.zeros((T, 3))
    imu_quaternion = np.zeros((T, 4))
    for i in range(T-1):
        r, p = calculate_roll_pitch(imu_observations[i, :3])
        # print("r.shape", r.shape)
        # print("type(r)", type(r))
        imu_rpy[i, 0] = r[0]
        imu_rpy[i, 1] = p[0]
        imu_rpy[i, 2] = 0

        imu_quaternion[i, :] = aa2quat(imu_rpy[i, :]).q
 


    '''PLOT'''
    
    rpy_fig, (roll_plot, pitch_plot, yaw_plot) = plt.subplots(3, 1)
    rpy_fig.suptitle("RPY Comparison: UKF v Calibrated IMU v Vicon")
    rpy_fig.tight_layout()

    roll_plot.set_title("Pitch [rad/s]")
    roll_plot.plot(time[:T], vicon_orientations_rpy[:T, 0], 
             label="roll_vicon", linestyle='solid', color='r')
    roll_plot.plot(time[:T], ukf_orientations_rpy[0][:T], 
             label="roll_ukf", linestyle='dotted', color='r', alpha=0.4)
    roll_plot.plot(time[:T], imu_rpy[:T, 0],
             label="roll_imu", linestyle='solid', color='r', alpha=0.4)
    roll_plot.set_xlabel("Time [s]")
    roll_plot.set_ylabel("Rotation [rad]")
    roll_plot.legend()
   
    pitch_plot.set_title("Pitch [rad/s]")
    pitch_plot.plot(time[:T], vicon_orientations_rpy[:T, 1], 
             label="pitch_vicon", linestyle='solid', color='g')
    pitch_plot.plot(time[:T], ukf_orientations_rpy[1][:T], 
             label="pitch_ufk", linestyle='dotted', color='g', alpha=0.4)
    pitch_plot.plot(time[:T], imu_rpy[:T, 1],
             label="pitch_imu", linestyle='solid', color='g', alpha=0.4)
    pitch_plot.set_xlabel("Time [s]")
    pitch_plot.set_ylabel("Rotation [rad]")
    pitch_plot.legend()
    
    yaw_plot.set_title("Yaw [rad/s]")
    yaw_plot.plot(time[:T], vicon_orientations_rpy[:T, 2], 
             label="yaw_vicon", linestyle='solid', color='b')
    yaw_plot.plot(time[:T], ukf_orientations_rpy[2][:T], 
             label="yaw_ufk", linestyle='dotted', color='b', alpha=0.4)
    yaw_plot.set_xlabel("Time [s]")
    yaw_plot.set_ylabel("Rotation [rad]")
    yaw_plot.legend()



    quat_fig, (ukf_quat_plot, imu_quat_plot, vicon_quat_plot) = plt.subplots(3, 1) 
    quat_fig.suptitle("Quat. Comparison: UKF v Calibrated IMU v Vicon")

    ukf_quat_plot.set_title("UKF")
    ukf_quat_plot.plot(time[:T], ukf_orientations[:T, 0],
             label="w", alpha=0.4)
    ukf_quat_plot.plot(time[:T], ukf_orientations[:T, 1],
             label="x", alpha=0.4)
    ukf_quat_plot.plot(time[:T], ukf_orientations[:T, 2],
             label="y", alpha=0.4)
    ukf_quat_plot.plot(time[:T], ukf_orientations[:T, 3],
             label="z", alpha=0.4)
    ukf_quat_plot.set_xlabel("Time [s]")
    ukf_quat_plot.set_ylabel("Quat. Component [-]")
    ukf_quat_plot.legend()

    imu_quat_plot.set_title("Calibrated IMU")
    imu_quat_plot.plot(time[:T], imu_quaternion[:T, 0],
             label="w", alpha=0.4)
    imu_quat_plot.plot(time[:T], imu_quaternion[:T, 1],
             label="x", alpha=0.4)
    imu_quat_plot.plot(time[:T], imu_quaternion[:T, 2],
             label="y", alpha=0.4)
    imu_quat_plot.plot(time[:T], imu_quaternion[:T, 3],
             label="z", alpha=0.4)
    imu_quat_plot.set_xlabel("Time [s]")
    imu_quat_plot.set_ylabel("Quat. Component [-]")
    imu_quat_plot.legend()

    vicon_quat_plot.set_title("Vicon")
    vicon_quat_plot.plot(time[:T], vicon_orientations[:T, 0],
             label="w", alpha=0.4)
    vicon_quat_plot.plot(time[:T], vicon_orientations[:T, 1],
             label="x", alpha=0.4)
    vicon_quat_plot.plot(time[:T], vicon_orientations[:T, 2],
             label="y", alpha=0.4)
    vicon_quat_plot.plot(time[:T], vicon_orientations[:T, 3],
             label="z", alpha=0.4)
    vicon_quat_plot.set_xlabel("Time [s]")
    vicon_quat_plot.set_ylabel("Quat. Component [-]")
    vicon_quat_plot.legend()



    quat_distribution_fig, (ukf_quat_mean_plot, ukf_quat_cov_plot, 
                            vicon_quat_mean_plot) = plt.subplots(3, 1)
    quat_distribution_fig.suptitle("Quaternion Distribution: UKF vs Vicon")
    
    ukf_quat_mean_plot.set_title("UKF Quaternion Mean")
    ukf_quat_mean_plot.plot(time[:T], ukf_orientations[:T, 0],
             label="w", alpha=0.4)
    ukf_quat_mean_plot.plot(time[:T], ukf_orientations[:T, 1],
             label="x", alpha=0.4)
    ukf_quat_mean_plot.plot(time[:T], ukf_orientations[:T, 2],
             label="y", alpha=0.4)
    ukf_quat_mean_plot.plot(time[:T], ukf_orientations[:T, 3],
             label="z", alpha=0.4)
    ukf_quat_mean_plot.set_xlabel("Time [s]")
    ukf_quat_mean_plot.set_ylabel("Quat. Component [-]")
    ukf_quat_mean_plot.legend()

    signs, abs_slogdets = np.linalg.slogdet(ukf_covariances[:T, :3, :3])
    quat_slogdets = signs * abs_slogdets

    ukf_quat_cov_plot.set_title("UKF Covariance Signed Log Determinant")
    ukf_quat_cov_plot.plot(time[:T], quat_slogdets,
             label='slogdet(cov)', alpha=0.4)
    ukf_quat_cov_plot.set_xlabel("Time [s]")
    ukf_quat_cov_plot.set_ylabel("Quat. Component [-]")
    ukf_quat_cov_plot.legend()

    vicon_quat_mean_plot.plot(time[:T], vicon_orientations[:T, 0],
             label="w", alpha=0.4)
    vicon_quat_mean_plot.plot(time[:T], vicon_orientations[:T, 1],
             label="x", alpha=0.4)
    vicon_quat_mean_plot.plot(time[:T], vicon_orientations[:T, 2],
             label="y", alpha=0.4)
    vicon_quat_mean_plot.plot(time[:T], vicon_orientations[:T, 3],
             label="z", alpha=0.4)
    vicon_quat_mean_plot.set_xlabel("Time [s]")
    vicon_quat_mean_plot.set_ylabel("Quat. Component [-]")
    vicon_quat_mean_plot.legend()



    omega_distribution_fig, (ukf_omega_mean_plot, ukf_omega_cov_plot, 
                             vicon_omega_mean_plot) = plt.subplots(3, 1)
    omega_distribution_fig.suptitle("Angular Velocity Distribution: UKF vs Vicon")

    ukf_omega_mean_plot.set_title("UKF Mean")
    ukf_omega_mean_plot.plot(time[:T], ukf_angular_velocities[:T, 0],
             label="wx", alpha=0.4)
    ukf_omega_mean_plot.plot(time[:T], ukf_angular_velocities[:T, 1],
             label="wy", alpha=0.4)
    ukf_omega_mean_plot.plot(time[:T], ukf_angular_velocities[:T, 2],
             label="wz", alpha=0.4)
    ukf_omega_mean_plot.set_xlabel("Time [s]", loc='right')
    ukf_omega_mean_plot.set_ylabel("Angular Velocity [rad/s]")
    ukf_omega_mean_plot.legend()

    ukf_omega_cov_plot.set_title("UKF Covariance Signed Log Determinant")
    signs, abs_slogdets = np.linalg.slogdet(ukf_covariances[:T, -3:, -3:])
    angular_velocity_slogdets = signs * abs_slogdets
    ukf_omega_cov_plot.plot(time[:T], angular_velocity_slogdets,
             label='slogdet(cov)', alpha=0.4)
    ukf_omega_cov_plot.set_xlabel("Time [s]")
    ukf_omega_cov_plot.set_ylabel("Angular Velocity [rad/s]")
    ukf_omega_cov_plot.legend()

    vicon_omega_mean_plot.set_title("Vicon Mean")
    vicon_omega_mean_plot.plot(time[:T], vicon_angular_velocities[:T, 0],
             label="wx", alpha=0.4)
    vicon_omega_mean_plot.plot(time[:T], vicon_angular_velocities[:T, 1],
             label="wy", alpha=0.4)
    vicon_omega_mean_plot.plot(time[:T], vicon_angular_velocities[:T, 2],
             label="wz", alpha=0.4)
    vicon_omega_mean_plot.set_xlabel("Time [s]")
    vicon_omega_mean_plot.set_ylabel("Angular Velocity [rad/s]")
    vicon_omega_mean_plot.legend()



    plt.show()







        
estimate_rot(1, enable_plot=True, recalibrate=False)



