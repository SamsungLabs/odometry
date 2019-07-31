import numpy as np
from pyquaternion import Quaternion

from slam.linalg_utils import convert_euler_angles_to_rotation_matrix, form_se3

def mean_trajectory(df):
    result = [row for row in df.values]
    se3_global_matrices={0: np.eye(4,4)}
    for i in result:
        dof = i
        from_global = se3_global_matrices.get(dof[0])
        new_global_matrix = from_global @ form_se3(convert_euler_angles_to_rotation_matrix(dof[2:5]), dof[5:])
        same_matrix = se3_global_matrices.get(dof[1])
        if same_matrix is not None:
            rsame, tsame = same_matrix[0:3,0:3], same_matrix[:3,3]
            rnew, tnew =  new_global_matrix[0:3, 0:3], new_global_matrix[:3, 3]
            qsame, qnew = Quaternion(matrix=rsame), Quaternion(matrix=rnew)
            r_slerp = Quaternion.slerp(qsame, qnew, 0.5)
            t_mean = np.array((tsame + tnew)/2)
            new_global_matrix = form_se3(r_slerp.rotation_matrix, t_mean)
        se3_global_matrices[dof[1]] = new_global_matrix
    return se3_global_matrices