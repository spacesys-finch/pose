from rotate_eci import *

import time

def test_rotate_z():
    r = np.random.rand(10, 3)
    target = np.random.rand(00, 3)

    z = np.array((0,0,1))
    z_i = np.repeat(z[np.newaxis, :], r.shape[0], axis=0)

    start = time.time()
    rot_mat = rotate_z(r, target)

    z_i_rot = np.einsum('bji,bi->bj', rot_mat, z_i)

    z_f = target - r
    z_f /= np.linalg.norm(z_f, axis=1)[:, np.newaxis]

    print(np.allclose(z_i_rot, z_f))

test_rotate_z()