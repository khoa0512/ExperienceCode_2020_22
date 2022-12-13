import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error



def load_data(holiday=False, pollutant=["PM2.5"]):

    # path to the csv file
    path = r'C:\Users\buing\Documents\school research things\topological data project\dmd\Hourly_data_of_Beijing_from_Jinxi_interpolated.csv'
    
    df = pd.read_csv(path)
    
    # Holiday includes weekend, weekday, holiday
    keys = ['ERA5_d2m', 'ERA5_t2m', 'ERA5_rh', 'ERA5_sp', 'ERA5_u10', 'ERA5_v10', 'ERA5_blh'] #, 'SO2', 'PM2.5', 'PM10', 'CO', 'NO2', 'O3']
    
    keys += pollutant
    if holiday:
        keys = ['Holiday'] + keys
    
    # we have 34 uniques sites
    sites = df['Site'].values
    unique_sites = np.unique(sites)

    # each site has data of 14 * 8760, first 12 are control variables, last 2 are labels
    x = []
    for unique_site in unique_sites:
        ind = sites == unique_site
        df_temp = df[keys].iloc[ind].interpolate().copy()
        if holiday:
            df_holiday = OneHotEncoder().fit_transform(df_temp.Holiday.values.reshape(-1, 1)).toarray()
            df_rest = df_temp.values[:, 1:].astype('float')
            df_temp = np.concatenate((df_holiday, df_rest), axis=1)
        x.append(df_temp)

    x = np.array(x).transpose((2, 0, 1))
    # -6 mean to the last sixth element
    u = x[:-len(pollutant), :] # --> :-6, from the first element to the last sixth
    
    x = x[-len(pollutant):, :] # --> -6:, the last six elements
    return x, u, unique_sites


def generate_data(x, u, window=96, size=None, rolling=False):
    x = x.T
    u = u.T 

    if rolling:
        if size == None:
            n = x.shape[-1] - window
        else:
            n = size
            
        x = np.array([x[..., i: i + window] for i in range(n)])
        u = np.array([u[..., i: i + window] for i in range(n)])
    else:
        if size == None:
            n = int(np.floor(x.shape[-1] / window))
        else:
            n = size
            
        x = np.array([x[..., i*window: (i+1)*window] for i in range(n)])
        u = np.array([u[..., i*window: (i+1)*window] for i in range(n)])
    return x, u


def error(x_true, x_pred, mode='percent'):
    x_true = x_true.ravel()
    x_pred = x_pred.ravel()

    if mode == 'percent':
        err1 = np.linalg.norm(x_true**2 - x_pred**2)
        err2 = np.linalg.norm(x_true**2)
        err = err1 / err2
    elif mode == 'rmse':
        err = mean_squared_error(x_true, x_pred) ** 0.5
    return err


def preprocess_data(x):
    x = np.concatenate([x[i] for i in range(x.shape[0])], axis=0).T
    # ss = StandardScaler()
    # xs = ss.fit_transform(x)

    # noise = np.random.RandomState(0).normal(loc=0, scale=0.05, size=xs.shape)
    # xs += noise
    return x

def dmd(x_train, x_test, i, s, n_train, n_test, r):
    n = n_train + n_test
    # select certain start time and site number
    if s == 'all':
        x_true = x_train[i, :, :, :]
        x_future = x_test[i, :, :, :]
        x_true = np.concatenate([x_true[:, i, :] for i in range(x_true.shape[1])], axis=0)
        x_future = np.concatenate([x_future[:, i, :] for i in range(x_future.shape[1])], axis=0)
    else:
        x_true = x_train[i, :, s, :]
        x_future = x_test[i, :, s, :]
        
    # define t and t+1 state variables
    x_0 = x_true[:, :-1]
    x_1 = x_true[:, 1:]

    ## Perform singular value decomposition on X1
    u, s, v = np.linalg.svd(x_0, full_matrices = False)
    ## Compute the Koopman matrix
    A_tilde = u[:, : r].conj().T @ x_1 @ v[: r, :].conj().T * np.reciprocal(s[: r])
    ## Perform eigenvalue decomposition on A_tilde
    Phi, Q = np.linalg.eig(A_tilde)
    ## Compute the coefficient matrix
    Psi = x_1 @ v[: r, :].conj().T @ np.diag(np.reciprocal(s[: r])) @ Q
    A = Psi @ np.diag(Phi) @ np.linalg.pinv(Psi)
    A = A.real
    # do prediction
    x_temp = x_0[:, 0]
    x_pred = [x_temp]
    for i in range(n):
        x_pred.append(A @ x_temp)
        x_temp = x_pred[i + 1]
    x_pred = np.array(x_pred).T
    x_recons = x_pred[:, :n_train]
    x_pred = x_pred[:, n_train:n_train + n_test]
    return x_true, x_recons, x_future, x_pred, A


def dmdtd(x_train, x_test, i, s, n_train, n_test, delay):
    n = n_train + n_test
    # select certain start time and site number
    if s == 'all':
        x_true = x_train[i, :, :, :]
        x_future = x_test[i, :, :, :]
        x_true = x_true.reshape(-1, x_true.shape[-1])
        x_future = x_future.reshape(-1, x_future.shape[-1])
    else:
        x_true = x_train[i, :, s, :]
        x_future = x_test[i, :, s, :]

    n_x = x_true.shape[0]

    # calculate delay embedding
    n_state, n_time = x_true.shape
    x_delay = np.zeros((n_x * delay, n_time - delay))

    for j in range(n_time - delay):
        for k in range(delay):
            x_delay[k * n_x: k * n_x + n_x, j] = x_true[:, j + k: j + k + 1].T

    # define t and t+1 state variables
    x_0 = x_delay[:, :-1]
    x_1 = x_delay[:, 1:]

    # calculate A matrix x_1 = Ax_0
    A = x_1 @ np.linalg.pinv(x_0)

    # do prediction
    x_temp = x_0[:, 0]
    x_pred = [x_temp]
    for i in range(n):
        x_pred.append(A @ x_temp)
        x_temp = x_pred[i + 1]
    x_pred = np.array(x_pred).T[-n_x:]  # select the last two values for the current time point
    x_true = x_true[:, delay - 1:]
    x_recons = x_pred[:, :n_train - delay + 1]
    x_pred = x_pred[:, n_train - delay + 1:n_train - delay + 1 + n_test]
    return x_true, x_recons, x_future, x_pred


def dmdc(x_train, x_test, u_train, u_test, i, s, n_train, n_test):
    n = n_train + n_test
    # select certain start time and site number
    if s == 'all':
        x_true = x_train[i, :, :, :]
        u_true = u_train[i, :, :, :]
        x_future = x_test[i, :, :, :]
        u_future = u_test[i, :, :, :]
        x_true = x_true.reshape(-1, x_true.shape[-1])
        x_future = x_future.reshape(-1, x_future.shape[-1])
        u_true = u_true.reshape(-1, u_true.shape[-1])
        u_future = u_future.reshape(-1, u_future.shape[-1])
    else:
        x_true = x_train[i, :, s, :]
        u_true = u_train[i, :, s, :]
        x_future = x_test[i, :, s, :]
        u_future = u_test[i, :, s, :]

    n_x = x_true.shape[0]

    # define t and t+1 state and control variables
    x_0 = x_true[:, :-1]
    x_1 = x_true[:, 1:]
    u_0 = u_true[:, :-1]
    u_1 = u_true[:, 1:]

    # calculate A, B matrix x_1 = Ax_0 + Bu_0 -- > x_1 = [A B] [x_0; u_0] --> x_1 = GΩ
    Omega = np.concatenate((x_0, u_0), axis=0)
    U, S, Vt = np.linalg.svd(Omega, full_matrices=False)

    # G = x_1 v s^-1 u^T
    # A = x_1 v s^-1 u1^T; B = x_1 v s^-1 u2^T
    U_x = U[:n_x, :]
    U_u = U[n_x:, :]
    A = x_1 @ Vt.T / S @ U_x.T
    B = x_1 @ Vt.T / S @ U_u.T

    # combine now and future u
    u_combine = np.concatenate((u_true, u_future), axis=1)
    # do prediction
    x_temp = x_0[:, 0]
    x_pred = [x_temp]
    for i in range(n):
        x_pred.append(A @ x_temp + B @ u_combine[:, i])
        x_temp = x_pred[i + 1]
    x_pred = np.array(x_pred).T

    x_recons = x_pred[:, :n_train]
    x_pred = x_pred[:, n_train:n_train + n_test]
    return x_true, x_recons, x_future, x_pred, B


def dmdctd(x_train, x_test, u_train, u_test, i, s, n_train, n_test, delay):
    n = n_train + n_test - delay
    # select certain start time and site number
    if s == 'all':
        x_true = x_train[i, :, :, :]
        u_true = u_train[i, :, :, :]
        x_future = x_test[i, :, :, :]
        u_future = u_test[i, :, :, :]
        x_true = x_true.reshape(-1, x_true.shape[-1])
        x_future = x_future.reshape(-1, x_future.shape[-1])
        u_true = u_true.reshape(-1, u_true.shape[-1])
        u_future = u_future.reshape(-1, u_future.shape[-1])
    else:
        x_true = x_train[i, :, s, :]
        u_true = u_train[i, :, s, :]
        x_future = x_test[i, :, s, :]
        u_future = u_test[i, :, s, :]

    n_x = x_true.shape[0]
    n_u = u_true.shape[0]

    # calculate delay embedding for variables
    n_time = x_true.shape[1]
    x_delay = np.zeros((n_x * delay, n_time - delay))
    u_delay = np.zeros((n_u * delay, n_time - delay))

    for j in range(n_time - delay):
        for k in range(delay):
            x_delay[k * n_x: k * n_x + n_x, j] = x_true[:, j + k: j + k + 1].T
            u_delay[k * n_u: k * n_u + n_u, j] = u_true[:, j + k: j + k + 1].T

    # define t and t+1 state variables
    x_0 = x_delay[:, :-1]
    x_1 = x_delay[:, 1:]
    u_0 = u_delay[:, :-1]
    u_1 = u_delay[:, 1:]

    # calculate A, B matrix x_1 = Ax_0 + Bu_0 -- > x_1 = [A B] [x_0; u_0] --> x_1 = GΩ
    Omega = np.concatenate((x_0, u_0), axis=0)
    U, S, Vt = np.linalg.svd(Omega, full_matrices=False)

    # G = x_1 v s^-1 u^T
    # A = x_1 v s^-1 u1^T; B = x_1 v s^-1 u2^T
    U_x = U[:n_x * delay, :]
    U_u = U[n_x * delay:, :]
    A = x_1 @ Vt.T / S @ U_x.T
    B = x_1 @ Vt.T / S @ U_u.T

    # combine now and future u
    u_combine = np.concatenate((u_true, u_future), axis=1)
    n_time = u_combine.shape[1]
    u_combine_delay = np.zeros((n_u * delay, n_time - delay))
    for j in range(n_time - delay):
        for k in range(delay):
            u_combine_delay[k * n_u: k * n_u + n_u, j] = u_combine[:, j + k: j + k + 1].T

    # do prediction
    x_temp = x_0[:, 0]
    x_pred = [x_temp]
    for i in range(n):
        x_pred.append(A @ x_temp + B @ u_combine_delay[:, i])
        x_temp = x_pred[i + 1]
    x_pred = np.array(x_pred).T[-2:]  # select the last two values for the current time point
    x_true = x_true[:, delay - 1:]
    x_recons = x_pred[:, :n_train - delay + 1]
    x_pred = x_pred[:, n_train - delay + 1:n_train - delay + 1 + n_test]
    return x_true, x_recons, x_future, x_pred, B
