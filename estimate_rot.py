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

def estimate_rot(data_num=1):
    #load data
    imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
    vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
    accel = imu['vals'][0:3,:]
    gyro = imu['vals'][3:6,:]
    rots = vicon["rots"]
    T = np.shape(imu['ts'])[1]

    T = min(T, accel.shape[1], gyro.shape[1], rots.shape[2])

    time_steps = np.arange(T)
    # your code goes here
   
    # print("T", T)
    # print("Gyro", gyro)
    # print("Gyro.shape", gyro.shape)
    # print("Accel", accel)
    # print("accel.shape", accel.shape)





    # print("Vicon.shape", vicon.shape)
    # print("vicon", vicon)

    # print("Rots.shape", rots.shape)
    # print("rots", rots)

    # Trim everything up to time T
    accel = accel[:, :T-1]
    gyro = gyro[:, :T-1]
    rots = rots[:, :, :T-1]

    # print("Trimed shapes Accel Gyro Rots:", accel.shape, gyro.shape, rots.shape)
    
    ## Rename variables for clarity

    # Trim cause only have T time steps
    # ARaw_B = accel[:, :T]
    # Transpose to fit my math better
    ARaw_B = accel.T
    # Append constant term to introduce bias in the math
    ones_column = np.ones((ARaw_B.shape[0], 1))
    ARaw_B = np.hstack((ARaw_B, ones_column))

    # print("aRaw_B.shape", aRaw_B.shape)
    # print("aRaw_B", aRaw_B)

    # Transpose each 3x3 (by swapping the first 2 dims)
    R_WB_transpose = np.transpose(rots, axes=(1, 0, 2))

    # Trim to T time
    # R_WB_transpose = R_WB_transpose[:, :, T]

    # Pad each 3x3 rot matrix into a 4x4
    # Specifically, add to the start and end of each dimension (0,1), (0,1), (0,1) many constant_values
    R_WB_transpose = np.pad(R_WB_transpose, ((0, 1), (0, 1), (0, 0)), mode='constant', constant_values=0)
    # Set bottom right term of each homogenized rotation matrix to 1
    R_WB_transpose[3, 3, :] = 1

    # print("R_WB_transpose.shape", R_WB_transpose.shape)
    # print("R_WB_transpose[:, :, 0]", R_WB_transpose[:, :, 0])


    # Create the stack of ground truth a_W vectors (just gravity) A_W
    A_W = np.zeros((T-1, 4, 1))
    A_W[:, 2, 0] = -9.81
    A_W[:, 3, 0] = 1

    
    # print("A_W.shape", A_W.shape)
    # print("A_W", A_W)
    
    # But actually numpy's batch dimension should be the first for both things
    R_WB_transpose = np.transpose(R_WB_transpose, axes=(2, 0, 1))

    Y = np.matmul(R_WB_transpose, A_W)

    Y = np.squeeze(Y)

    # print("Y.shape", Y.shape)
    # print("Y", Y)


    X = np.squeeze(ARaw_B)


    # print("X.shape", X.shape)
    

    K = (np.linalg.inv(X.T @ X) @ X.T @ Y).T

    np.set_printoptions(precision=4, suppress=True)

    # print("K.shape", K.shape)
    # print("K", K)



    ## Now solve for accel alpha and beta

    accel_alpha = np.zeros(3)
    accel_beta = np.zeros(3)

    for row in range(K.shape[0]-1):
        k_i = K[row][row]
        b_i = K[row][-1]

        accel_alpha[row] = 3300 / (1023 * k_i)
        accel_beta[row] = - b_i / k_i
        # print("k_i", k_i)
        # print("b_i", b_i)
        # print("accel_alpha[row]", accel_alpha[row])
        # print("accel_beta[row]", accel_beta[row])

    print("accel_alpha: ", accel_alpha)
    print("accel_beta: ", accel_beta)
 

    # Solve with the closed form solution to linear regression
    # w = np.linalg.inv(X.T @ X) @ X.T @ Y

    # roll, pitch, yaw are numpy arrays of length T
    # return roll,pitch,yaw



    ## Plotting

    # first, plot the raw data
    # plt.figure()
    # for i in range(3):
    #     plt.subplot(2, 1, 1)
    #     plt.plot(time_steps, accel[i])
    #     plt.subplot(2, 1, 2)
    #     plt.plot(time_steps, gyro[i])
    #
    # plt.subplot(2, 1, 1)
    # plt.legend(["a_x", "a_y", "a_z"])
    #
    # plt.subplot(2, 1, 2)
    # plt.legend(["omega_x", "omega_y", "omega_z"])
    # 
    #
    # plt.show()

# estimate_rot(1)
