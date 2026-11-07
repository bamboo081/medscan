import h5py
import numpy as np
from convert_figshare_mri import convert_one


def test_figshare_mat_conversion(tmp_path):
    path = tmp_path / '1.mat'
    with h5py.File(path, 'w') as f:
        c = f.create_group('cjdata')
        c.create_dataset('image', data=np.arange(16,dtype=np.uint16).reshape(4,4))
        c.create_dataset('label', data=[[2]])
        c.create_dataset('PID', data=[[ord(x)] for x in 'patient0001'])
    image, pid, label = convert_one(path)
    assert label == 'glioma' and pid == 'patient0001' and image.shape == (4,4)
