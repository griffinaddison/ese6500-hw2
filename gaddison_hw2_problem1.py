import numpy as np

import matplotlib.pyplot as plt


## Part 1: Simulate and create dataset

# Create time vector of 100 steps, defaulting from 0, increments of 1
total_steps = 100
time_steps = np.arange(total_steps)
a = -1

# Initialize state and observation trajectories
x = np.zeros(total_steps)
y = np.zeros(total_steps)


# Set initial state
x[0] = np.random.normal(1, np.sqrt(2)) # takes stddev as 2nd argument, not variance


# For each time step
for k in range(total_steps-1):

    # Calculate state noise
    epsilon_k = np.random.normal(0, np.sqrt(1))

    # Calculate next state
    x[k+1] = a * x[k] + epsilon_k


    # Calculate observation noise
    eta_k = np.random.normal(0, np.sqrt(1/2))

    # Calculate observation
    y[k+1] = np.sqrt(x[k+1]**2 + 1) + eta_k






## Part 2: Estimate parameter a using Extended Kalman Filter


# Define helper functions

def h(z):
    [x], [a] = z
    return np.array([[a * x], [a]])


def g(z):
    [x], _ = z
    return np.sqrt(x**2 + 1)

def A(z): # Jacobian of f
    [x], [a] = z
    return np.array([[a, x],
                     [0, 1]])

def C(z): # Jacobian of g
    [x], [a] = z    
    return np.array([(x / np.sqrt(x**2 + 1)), 0]).reshape(1, 2)





# Define our parameter estimate trajectories (mean and covariance)
mu_est = np.zeros((total_steps, 2, 1))
sigma_est = np.zeros((total_steps, 2, 2))

# Initial conditions
x_0 = np.random.normal(1, np.sqrt(2))
a_0 = -0.5 # Set arbitrary initial guess for a
mu_est[0] = np.array([x_0, a_0]).reshape(-1, 1)
var_x = 1
var_a = 6
sigma_est[0] = np.array([[var_x, 0],[0, var_a]])



## Define noise covariances
# For state x noise 
sigma_sq_k = 1;
sigma_sq_a_k = 0 # We know a constant => var(a) = 0
R = np.array([[sigma_sq_k,  0], 
              [0,           sigma_sq_a_k]])
# For observation y noise
Q = np.array([[1/2]])

I = np.eye(2)  # Identity matrix


# Loop over each time step
for k in range(total_steps - 1):

    # Rename variables for clarity
    mu_kgk = mu_est[k].reshape(2, 1)
    sigma_kgk = sigma_est[k]
 
    # Predict next state and variance
    mu_kp1gk = h(mu_kgk)    
    sigma_kp1gk = A(mu_kgk) @ sigma_kgk @ A(mu_kgk).T + R


    
    # Rename variables for clarity 
    y_kp1 = y[k+1]

    # Update estimates based on observation
    K = sigma_kp1gk @ C(mu_kp1gk).T \
        @ np.linalg.inv(C(mu_kp1gk) @ sigma_kp1gk @ C(mu_kp1gk).T + Q)
    mu_kp1gkp1 = mu_kp1gk + K * (y_kp1 - g(mu_kp1gk))
    sigma_kp1gkp1 = (I - K @ C(mu_kp1gk)) @ sigma_kp1gk

    # Rename variables for clarity
    mu_est[k+1] = mu_kp1gkp1.reshape(-1, 1)
    sigma_est[k+1] = sigma_kp1gkp1
    

   
    

# Plotting the EKF results
plt.figure(figsize=(12, 8))

# Plotting the state estimates
plt.subplot(2, 1, 1)
plt.plot(time_steps, x, label='True State', color='blue')
plt.plot(time_steps, mu_est[:, 0], label='Estimated State', color='orange', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('State')
plt.title('State Trajectory')
plt.legend()

# Plotting the parameter 'a' estimates
plt.subplot(2, 1, 2)
plt.plot(time_steps, np.full(total_steps, a), label='True a', color='red')
plt.plot(time_steps, mu_est[:, 1], label='Estimated a', color='green', linestyle='--')
mu_est_max = mu_est[:, 1, 0] + sigma_est[:, 1, 0]
mu_est_min = mu_est[:, 1, 0] - sigma_est[:, 1, 0]
plt.fill_between(time_steps, mu_est_max, mu_est_min, color='green', alpha=0.2)
plt.xlabel('Time Step')

plt.xlabel('Time Step')
plt.ylabel('Parameter a')
plt.title('Parameter Estimation Trajectory')
plt.legend()

plt.tight_layout()
plt.show()

