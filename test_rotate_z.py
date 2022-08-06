from rotate_z import *

def test_rotate_z():
    #r = np.random.rand(10, 3)
    #target = np.random.rand(10, 3)

    r = np.array([[1.,1.,0.],[0.,1.,1.]])/(2**0.5)
    target = np.array([[0.,0.,0.],[0.,0.,0.]])

    z = np.array((0,0,1))
    z_i = np.repeat(z[np.newaxis, :], r.shape[0], axis=0)

    rot_mat = rotate_z(r, target)

    z_i_rot = np.einsum('bji,bi->bj', rot_mat, z_i)

    z_f = target - r

    print((z_i_rot == z_f).all())

test_rotate_z()