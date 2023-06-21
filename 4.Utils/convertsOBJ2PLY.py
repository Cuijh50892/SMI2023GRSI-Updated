import os
import math
import json
import argparse
import numpy as np
import pyntcloud


pymesh_path = "../TestParts/1/1_part_1"
point_cloud_size = 2049

pts_path = pymesh_path + '.pts'
ply_path = pymesh_path + '_2048' + ".ply"
pynt = pyntcloud.PyntCloud.from_file(pymesh_path + ".obj")
cloud = pynt.get_sample('mesh_random', n=point_cloud_size)
cloud = cloud.values

for dim in [0, 1, 2]:
    dim_mean = np.mean(cloud[:, dim])
    cloud[:, dim] -= dim_mean

# Scale to unit-ball
distances = [np.linalg.norm(point) for point in cloud]
scale = 1. / np.max(distances)
cloud *= scale
np.savetxt(pts_path, cloud, fmt='%0.8f')
plyout = pyntcloud.PyntCloud.from_file(pts_path, sep=" ", header=0, names=["x", "y", "z"])
plyout.to_file(ply_path)











