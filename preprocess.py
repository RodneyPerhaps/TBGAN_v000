import os
import os.path as osp
import menpo
import menpo3d
import numpy as np
import menpo.io as mio
import menpo3d.io as m3io
from menpo.shape import TexturedTriMesh, TriMesh, ColouredTriMesh
import scipy.io as sio

lsfm_model_path = "/home/jschen/wilduv_model"
shape_path = osp.join(lsfm_model_path, 'shape_nicp')
texture_path = osp.join(lsfm_model_path, 'texture')
shape_model_fp = osp.join(lsfm_model_path, 'shape_model.mat')

shapeUV_path = osp.join(lsfm_model_path, 'shapeUV')
normalUV_path = osp.join(lsfm_model_path, 'normalUV')

trilist = sio.loadmat(shape_model_fp)['trilist']
lsfm_model = m3io.import_lsfm_model(shape_model_fp)

def normalize0to1(points):
    min = np.array([points[:,0].min(), shape[:,1].min(),shape[:,2].min()])
    max = np.array([points[:,0].max(),shape[:,1].max(),shape[:,2].max()])
    return (points-min)/(max-min)


for shape_fp in os.listdir(shape_path):
    shape = mio.import_pickle(osp.join(shape_path, shape_fp)).reshape([-1, 3])
    normalized_shape = normalize0to1(shape)
    mesh = TriMesh(shape, trilist)
    normal = mesh.vertex_normals()
    normalized_normal = normalize0to1(normal)

    shape_mesh = ColouredTriMesh(lsfm_model.mean_vector.reshape([-1, 3]), trilist, normalized_shape)
    normal_mesh = ColouredTriMesh(lsfm_model.mean_vector.reshape([-1, 3]), trilist, normalized_normal)
    print('tst')


