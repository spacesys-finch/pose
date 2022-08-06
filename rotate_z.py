import numpy as np

def get_rotation(a, b):
    # Rotation matrix which rotates a to b about the normal of the plane they form, after rotating the normal to the z axis
    # cos(theta) == dot(a, b)
    cos = np.einsum('ij,ij->i', a, b) 

    # sin(theta) == |cross(A, B)|
    cross = np.cross(a, b)
    sin = np.linalg.norm(cross, axis=1) 
    cross /= sin[:, np.newaxis]
    
    zeros = np.zeros(a.shape[0])
    ones = np.ones(a.shape[0])

    R_w_x = np.stack((cos, -sin, zeros), axis=1)
    R_w_y = np.stack((sin, cos, zeros), axis=1)
    R_w_z = np.stack((zeros, zeros, ones), axis=1)

    R_w = np.stack((R_w_x, R_w_y, R_w_z), axis=1)

    # basis such that w is normal to z_i and z_f
    u = a
    v = b - cos@a
    v /= np.linalg.norm(v, axis=1)[:, np.newaxis]
    if v.ndim == 1:
        v = v[np.newaxis, :]
    #w = cross

    # change of basis matrices from xyz->uvw and uvw->xyz respectively
    C_inv = np.stack((u, v, cross), axis=2)
    C = np.linalg.inv(C_inv)

    return C_inv @ R_w @ C

def rotate_z(r, target):
    # z after rotation
    z_f = target - r

    # normalization
    z_f /= np.linalg.norm(z_f, axis=1)[:, np.newaxis]

    z = np.array((0,0,1))
    z_i = np.repeat(z[np.newaxis, :], r.shape[0], axis=0)

    return get_rotation(z_i, z_f)

def rotate_xy(r, target, across):
    # z after rotation
    z_f = target - r

    # x after rotation is normal to plane formed by z_f and across vector
    x_f = np.cross(z_f, across)

    # normalization
    x_f /= np.linalg.norm(x, axis=1)[:, np.newaxis]

    x = np.array((1,0,0))
    x_i = np.repeat(x[np.newaxis, :], r.shape[0], axis=0)

    return get_rotation(x_i, x_f)