import menpo
import menpo3d
import numpy as np
import menpo.io as mio
import menpo3d.io as m3io
from menpo.shape import TexturedTriMesh, TriMesh, ColouredTriMesh
import scipy.io as sio

def from_UV_2_3D():
    pass

def from_3D_2_UV(p):
    """input: vertices: n(x,y,z) x 3
    return 3x256x256
    """
    import numpy as np

    p = np.random.rand(53215,3)
    min = np.array([p[:,0].min(), p[:,1].min(),p[:,2].min()])
    max = np.array([p[:,0].max(),p[:,1].max(),p[:,2].max()])
    q = 2.0*(p-min)/(max-min)-1.0

    tcoords = mio.import_pickle('/home/jschen/TBGAN_v000/tcoord.pkl')
    tcoords = tcoords * 256
    

def UV_tex_2_UV():
    pass