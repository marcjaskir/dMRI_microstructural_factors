"""Handle merging and spliting of DSI files."""
import numpy as np
import os
import os.path as op
import nibabel as nb
from dipy.core.geometry import cart2sphere
from dipy.core.sphere import HemiSphere
from dipy.direction import peak_directions
import subprocess
from scipy.io.matlab import loadmat, savemat
from tqdm import tqdm
import re
import gzip
import tempfile
import shutil
ODF_COLS = 20000  # Number of columns in DSI Studio odf split


def popen_run(arg_list):
    cmd = subprocess.Popen(arg_list, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    out, err = cmd.communicate()
    print(out)
    print(err)


def dsistudio_to_mrtrix(fib_file, source_nifti_file, sh_mif):
    """Convert a DSI Studio fib file to mrtrix.

    The fib_file can be compressed (.fib.gz) or uncompressed (.fib). If
    compressed, it will be automatically decompressed. Actual affine information
    is not stored in the fib file, so you need to supply a nifti image that has
    a usable sform or qform matrix stored in the header. This nifti must also
    be in LPS+ orientation.

    The output is a MRTrix ``.mif`` file that will contain sh coefficients
    that represent the ODF calculated in DSI Studio. **NOTE:** the ODF created
    by GQI is much less sharp than those created by CSD. They will look like
    blobs. They don't look like this in DSI Studio because the minimum value
    is subtracted before they are rendered as glyphs. If you run ``fod2fixel``
    on the output from this function, you will see that it still gets all the
    crossings and the orientations are correct.

    Parameters:
    ============

    fib_file : str
        Path to a fib file (can be .fib or .fib.gz)
    source_nifti_file : str
        Path to a nifti file with correct spatial mapping
    sh_mif : str
        File name for the MRTrix mif file that will be created.

    """
    # Handle compressed files
    temp_fib = None
    if fib_file.endswith('.gz'):
        # Decompress using subprocess for speed (as noted in original comment)
        temp_fib = tempfile.NamedTemporaryFile(delete=False, suffix='.fib')
        temp_fib.close()
        try:
            # Use gunzip for faster decompression
            with open(temp_fib.name, 'wb') as f_out:
                result = subprocess.run(['gunzip', '-c', fib_file], 
                                      stdout=f_out,
                                      stderr=subprocess.PIPE, check=True)
            fib_file_to_load = temp_fib.name
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to Python gzip if gunzip not available
            with gzip.open(fib_file, 'rb') as f_in:
                with open(temp_fib.name, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            fib_file_to_load = temp_fib.name
    else:
        fib_file_to_load = fib_file
    
    try:
        fibmat = loadmat(fib_file_to_load)
    except Exception as e:
        if temp_fib and os.path.exists(temp_fib.name):
            os.remove(temp_fib.name)
        raise ValueError(f"Failed to load MATLAB file {fib_file}: {e}")
    
    # Clean up temporary file after successful load (loadmat reads entire file into memory)
    if temp_fib and os.path.exists(temp_fib.name):
        os.remove(temp_fib.name)
    dims = tuple(fibmat['dimension'].squeeze())
    directions = fibmat['odf_vertices'].T

    odf_vars = [k for k in fibmat.keys() if re.match("odf\\d+", k)]

    valid_odfs = []
    flat_mask = (fibmat["fa0"].squeeze() > 0).flatten()
    n_voxels = np.prod(dims)
    for n in range(len(odf_vars)):
        varname = "odf%d" % n
        odfs = fibmat[varname]
        odf_sum = odfs.sum(0)
        odf_sum_mask = odf_sum > 0
        valid_odfs.append(odfs[:, odf_sum_mask].T)
    odf_array = np.row_stack(valid_odfs)
    odf_array = odf_array - odf_array.min(0)

    # Convert each column to a 3d file, then concatenate them
    odfs_3d = []
    for odf_vals in odf_array.T:
        new_data = np.zeros(n_voxels, dtype=np.float32)
        # Ensure odf_vals is 1D and has the right length
        odf_vals = odf_vals.flatten()
        n_valid = flat_mask.sum()
        if len(odf_vals) != n_valid:
            raise ValueError(f"Shape mismatch: odf_vals has {len(odf_vals)} values but flat_mask has {n_valid} True values")
        new_data[flat_mask] = odf_vals
        odfs_3d.append(new_data.reshape(dims, order="F"))

    real_img = nb.load(source_nifti_file)
    odf4d = np.stack(odfs_3d, -1)
    odf4d_img = nb.Nifti1Image(odf4d, real_img.affine, real_img.header)
    odf4d_img.to_filename("odf_values.nii")

    num_dirs, _ = directions.shape
    hemisphere = num_dirs // 2
    x, y, z = directions[:hemisphere].T
    _, theta, phi = cart2sphere(-x, -y, z)
    dirs_txt = op.join(os.getcwd(), "ras+directions.txt")
    np.savetxt(dirs_txt, np.column_stack([phi, theta]))

    popen_run(["amp2sh", "-force", "-directions", dirs_txt, "odf_values.nii", sh_mif])
    os.remove("odf_values.nii")
