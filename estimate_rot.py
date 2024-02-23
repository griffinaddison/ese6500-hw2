import numpy as np
from scipy import io
from quaternion import Quaternion
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



def estimate_rot(data_num=1):
    #load data
    imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
    vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
    accel = imu['vals'][0:3,:].astype(float)
    gyro = imu['vals'][3:6,:].astype(float)
    rots = vicon["rots"].astype(float)
    T = np.shape(imu['ts'])[1]

    print("imported gyro", gyro)

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


    print("gyro before\n", gyro)

    # Reorder the gyro readings (cuz they come in as Z, X, Y for some reason) 
    gyro = gyro[[1, 2, 0], :]


    # gyro = gyro[[0, 2, 1], :]

    # Make the unsigned ints into ints to allow negatives
    # gyro = gyro.astype(np.int16)


    # Negate x and y aces (cuz this imu is special)
    
    # gyro[0, :] = -gyro[0, :]
    # gyro[1, :] = -gyro[1, :]

    print("gyro after\n", gyro)

    ## Assemble the vector of raw accels in body frame

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
    A_W[:, 2, 0] = -9.81
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

    print("accel_alpha: ", accel_alpha)
    print("accel_beta: ", accel_beta)
 
    



    ### Calibrating the gyroscope



    # print("time.shape", time.shape)
    # print("time", time)

    omegaRaw_B = gyro.T

    omegaRaw_B = np.pad(omegaRaw_B, ((0, 0), (0, 1)), mode='constant', constant_values=1)

    # print("omegaRaw_B[0]", omegaRaw_B[0])
    
    omegaRaw_B = omegaRaw_B.reshape(-1, 1, 4)

    # print("omegaRaw_B.shape", omegaRaw_B.shape)


    # print("omegaRaw_B[0]", omegaRaw_B[0])

    
    R_WB = np.transpose(R_WB_transpose, axes=(0, 2, 1))

    # omegaRaw_W = omegaRaw_B @ R_WB_transpose
    #
    # omegaRaw_W = np.squeeze(omegaRaw_W)
    #
    # print("omegaRaw_W.shape", omegaRaw_W.shape)


    # print("omega_raw.shape", omega_raw.shape)
    # print("omega_raw", omega_raw)


    # print("R_WB_transpose.shape", R_WB_transpose.shape)
    # print("R_WB_transpose[0]\n", R_WB_transpose[0])


    # print("R_WB[0]\n", R_WB[0])


    omegaTrue_B = np.zeros((T-1, 1, 4))

    for i in range(T-2):

        R_i = R_WB[i]

        dt = time[i+1] - time[i]
        
        Rdot_i = (R_WB[i+1] - R_WB[i]) / dt

        # Rdot_i = (R_WB[i+1] @ R_WB[i].T) / dt

        # print("dt\n", dt)

        # print("np.linalg.inv(R_i)\n", np.linalg.inv(R_i))
        # print("")
        omegaTrue_skew_i = (np.linalg.inv(R_i) @ Rdot_i)

        # print("omegaTrue_skew_i\n", omegaTrue_skew_i)

        omegaTrue_W = np.hstack((unskew(omegaTrue_skew_i), [[1]]))

        omegaTrue_B[i] = omegaTrue_W @ R_WB[i+1]


        # print("omegaTrue_W\n", omegaTrue_W)
        # print("omegaTrue_B\n", omegaTrue_B)


    omegaTrue_B = np.squeeze(omegaTrue_B)


    # Trim last time step off, because omega_true is all 0
    omegaTrue_B = omegaTrue_B[:-1, :]
    omegaRaw_B = omegaRaw_B[:-1, :]

    # print("omega_true.shape", omega_true.shape)
    # print("omega_true\n", omega_true)
    # print("omega_raw.shape", omega_raw.shape)
    # print("omega_raw\n", omega_raw)

    X = np.squeeze(omegaRaw_B)
    Y = omegaTrue_B

    # print("X.shape", X.shape)
    # print("Y.shape", Y.shape)


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

    print("gyro_alpha: ", gyro_alpha)
    print("gyro_beta: ", gyro_beta)
 
    



    



    # # omega_true_skew_i = Rdot_i @ np.linalg.inv(R_i)




    # roll, pitch, yaw are numpy arrays of length T
    # return roll,pitch,yaw



    ## Plotting

    # first, plot the raw data
    # plt.figure()
    # for i in range(3):
    #     plt.subplot(2, 1, 1)
    #     plt.plot(time, accel[i])
    #     plt.subplot(2, 1, 2)
    #     plt.plot(time, gyro[i])
    #
    # plt.subplot(2, 1, 1)
    # plt.legend(["a_x", "a_y", "a_z"])
    #
    # plt.subplot(2, 1, 2)
    # plt.legend(["omega_x", "omega_y", "omega_z"])
    # 
    #
    # plt.show()

estimate_rot(1)
