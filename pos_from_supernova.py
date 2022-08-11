import numpy as np
from scipy.spatial.transform import Rotation as R

# loads data
positions = np.load("supernova_data/ECI_positions.npy").T
t = np.load("supernova_data/timesteps.npy")
windows = np.load("supernova_data/imaging_windows.npy", allow_pickle=True)

# splitting data into list of passes 
pos_ls = [None]*(windows.shape[0])
along_ls = [None]*(windows.shape[0])

for i, window in enumerate(windows):
    i_0 = window[0][0]
    i_n = window[0][-1]

    # initial position if it is not provided
    if i == 0 and i_0 == 0:
        # uses time step between t[0] and t[1] for initial time
        delta_t = t[1]-t[0]
        # use velocity at t[1]
        v_init = (positions[:,2] - positions[:,0])/(t[2] - t[0])
        # calculates initial position from velocity and time
        pos_init = positions[:,2] - v_init*delta_t

        pos_ls[0] = np.insert(positions[:, np.arange(i_0, i_n+1)], 0, pos_init, axis=1)
    
    # final position if it is not provided
    elif i == windows.shape[0]-1 and i_n == positions.shape[1]:
        # uses time step between t[-1] and t[-2] for initial time
        delta_t = t[-1]-t[-2]
        # use velocity at t[-2]
        v_f = (positions[:,-1] - positions[:,-3])/(t[-1] - t[-3])
        # calculates initial position from velocity and time
        pos_f = positions[:,-1] + v_f*delta_t
        
        pos_ls[-1] = np.insert(positions[:, np.arange(i_0, i_n+1)], 0, pos_f, axis=1)

    else:
        pos_ls[i] = positions[:, np.arange(i_0-1, i_n+1)]

    # normalizes position vector
    pos_ls[i] = -pos_ls[i]/np.linalg.norm(pos_ls[i], axis=0)

    along_ls[i] = np.diff(pos_ls[i])