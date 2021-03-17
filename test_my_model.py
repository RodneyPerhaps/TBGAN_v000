import os
import time
import re
import bisect
import numpy as np
import tensorflow as tf
import scipy.ndimage
import scipy.misc
from scipy.spatial.distance import cdist

import config
import misc
import tfutil
import myutil
import menpo.io as mio
import menpo3d.io as m3io
from menpo.shape import TexturedTriMesh, TriMesh, ColouredTriMesh
#from UV_spaces_V2.UV_manipulation_2 import from_UV_2_3D, from_3D_2_UV, UV_tex_2_UV
from menpo.image import Image

def generate_fake_images(run_id, snapshot=None, grid_size=[1,1],batch_size=8, num_pngs=1, image_shrink=1, png_prefix=None, random_seed=1000, minibatch_size=8):
    network_pkl = misc.locate_network_pkl(run_id, snapshot)
    if png_prefix is None:
        png_prefix = misc.get_id_string_for_network_pkl(network_pkl) + '-'
    random_state = np.random.RandomState(random_seed)

    print('Loading network from "%s"...' % network_pkl)
    G, D, Gs = misc.load_network_pkl(run_id, snapshot)

    lsfm_model = m3io.import_lsfm_model('/home/jschen/wilduv_model/shape_model.mat')
    lsfm_tcoords = \
    mio.import_pickle('/home/baris/Projects/team members/stelios/UV_spaces_V2/UV_dicts/full_face/512_UV_dict.pkl')[
        'tcoords']
    lsfm_params = []

    result_subdir = misc.create_result_subdir(config.result_dir, config.desc)
    for png_idx in range(int(num_pngs/batch_size)):
        start = time.time()
        print('Generating png %d-%d / %d... in ' % (png_idx*batch_size,(png_idx+1)*batch_size, num_pngs),end='')
        latents = misc.random_latents(np.prod(grid_size)*batch_size, Gs, random_state=random_state)
        labels = np.zeros([latents.shape[0], 0], np.float32)
        images = Gs.run(latents, labels, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_shrink=image_shrink)
        for i in range(batch_size):
            if images.shape[1]==3:
                mio.export_pickle(images[i],os.path.join(result_subdir, '%s%06d.pkl' % (png_prefix, png_idx*batch_size+i)))
                # misc.save_image(images[i], os.path.join(result_subdir, '%s%06d.png' % (png_prefix, png_idx*batch_size+i)), [0,255], grid_size)
            elif images.shape[1]==6:
                mio.export_pickle(images[i][3:6],
                                  os.path.join(result_subdir, '%s%06d.pkl' % (png_prefix, png_idx * batch_size + i)),overwrite=True)
                misc.save_image(images[i][0:3], os.path.join(result_subdir, '%s%06d.png' % (png_prefix, png_idx*batch_size+i)), [-1,1], grid_size)
            elif images.shape[1]==9:
                texture = Image(np.clip(images[i, 0:3] / 2 + 0.5, 0, 1))
                mesh_raw = from_UV_2_3D(Image(images[i, 3:6]))
                normals = images[i, 6:9]
                normals_norm = (normals - normals.min()) / (normals.max() - normals.min())
                mesh = lsfm_model.reconstruct(mesh_raw)
                lsfm_params.append(lsfm_model.project(mesh_raw))
                t_mesh = TexturedTriMesh(mesh.points, lsfm_tcoords.points, texture, mesh.trilist)
                m3io.export_textured_mesh(t_mesh,
                                          os.path.join(result_subdir, '%06d.obj' % (png_idx * minibatch_size + i)),
                                          texture_extension='.png')
                mio.export_image(Image(normals_norm),
                                 os.path.join(result_subdir, '%06d_nor.png' % (png_idx * minibatch_size + i)))
                shape = images[i, 3:6]
                shape_norm = (shape - shape.min()) / (shape.max() - shape.min())
                mio.export_image(Image(shape_norm),
                                 os.path.join(result_subdir, '%06d_shp.png' % (png_idx * minibatch_size + i)))
                mio.export_pickle(t_mesh,os.path.join(result_subdir, '%06d.pkl'% (png_idx * minibatch_size + i)))

        print('%0.2f seconds' % (time.time() - start))

    open(os.path.join(result_subdir, '_done.txt'), 'wt').close()


def unwrap():
    lsfm_model = m3io.import_lsfm_model('/home/jschen/wilduv_model/shape_model.mat')
    