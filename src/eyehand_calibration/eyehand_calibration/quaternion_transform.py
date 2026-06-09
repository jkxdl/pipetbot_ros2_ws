import numpy as np
import tf_transformations

# 标定结果
rotation_matrix = np.array([
    [-0.32748104,-0.07751873,0.94167246],
    [-0.94241266,-0.04485462,-0.3314309],
    [0.06793046,-0.99598138,-0.0583656]
])

translation_vector = np.array([-0.02946836,0.59097235,0.30800321])

# 转换旋转矩阵为四元数
quaternion = tf_transformations.quaternion_from_matrix(np.vstack([
    np.hstack([rotation_matrix, [[0], [0], [0]]]),
    [0, 0, 0, 1]
]))

# 打印结果
print("Translation:", translation_vector)
print("Quaternion:", quaternion)
