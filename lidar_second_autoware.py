#-*- coding: UTF-8 -*-

import os
import rospy
import math
import sys


from glob import glob

from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2
from autoware_msgs.msg import DetectedObject,DetectedObjectArray
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

# from msg import DetectedObject, DetectedObjectArray
# sys.path.remove('/home/ogailab/tiatia/codes/catkin_tia/src/lidar_second/')

import numpy as np
import pickle
from pathlib import Path
import pykitti
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import second.core.box_np_ops as box_np_ops

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

import torch
from google.protobuf import text_format
from second.utils import simplevis
from second.pytorch.train import build_network
from second.protos import pipeline_pb2
from second.utils import config_tool
import time

point_size = 1.0
axes_str = ['X', 'Y', 'Z']
axes_limits = [
    [-20, 80], # X axis range
    [-20, 20], # Y axis range
    [-3, 10]   # Z axis range
]

'''
相比1.0,改进：
1. 用class的方式sub和pub物体
2. 增加计时功能
3. 保存了rviz的config文件

further：
1. 用bbox画框，
2. 直接画在velodyne point上，而无需重新pub一个pcl2的点云（能节省80ms）

'''

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}
# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    """Return quaternion from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    >>> q = quaternion_from_euler(1, 2, 3, 'ryxz')
    >>> np.allclose(q, [0.310622, -0.718287, 0.444435, 0.435953])
    True

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    # print("ak : {}".format(type(ak)))
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    quaternion = np.empty((4,), dtype=np.float64)
    if repetition:
        quaternion[i] = cj * (cs + sc)
        quaternion[j] = sj * (cc + ss)
        quaternion[k] = sj * (cs - sc)
        quaternion[3] = cj * (cc - ss)
    else:
        quaternion[i] = cj * sc - sj * cs
        quaternion[j] = cj * ss + sj * cc
        quaternion[k] = cj * cs - sj * sc
        quaternion[3] = cj * cc + sj * ss
    if parity:
        quaternion[j] *= -1

    return quaternion

def bounding_box(points, min_x=-np.inf, max_x=np.inf, min_y=-np.inf,
                        max_y=np.inf, min_z=-np.inf, max_z=np.inf):
    """ Compute a bounding_box filter on the given points

    Parameters
    ----------
    points: (n,3) array
        The array containing all the points's coordinates. Expected format:
            array([
                [x1,y1,z1],
                ...,
                [xn,yn,zn]])

    min_i, max_i: float
        The bounding box limits for each coordinate. If some limits are missing,
        the default values are -infinite for the min_i and infinite for the max_i.

    Returns
    -------
    bb_filter : boolean array
        The boolean mask indicating wherever a point should be keeped or not.
        The size of the boolean mask will be the same as the number of given points.

    """

    bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
    bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)
    bound_z = np.logical_and(points[:, 2] > min_z, points[:, 2] < max_z)

    bb_filter = np.logical_and(bound_x, bound_y, bound_z)

    return bb_filter

def second_inference(header, points, measure_time):

    t0 = time.time()
    m = voxel_generator.generate(points, max_voxels=60000)
    voxels = m['voxels']
    coords = m['coordinates']
    num_points = m['num_points_per_voxel']

    # add batch idx to coords
    coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)
    voxels = torch.tensor(voxels, dtype=torch.float32, device=device)
    coords = torch.tensor(coords, dtype=torch.int32, device=device)
    num_points = torch.tensor(num_points, dtype=torch.int32, device=device)

    # Detection
    example = {
        "anchors": anchors,
        "voxels": voxels,
        "num_points": num_points,
        "coordinates": coords,
    }

    t1 = time.time()

    pred = net(example)[0]
    # print(pred)
    t2 = time.time()

    # Simple Vis-- color the points of detected object
    # pred = pred

    boxes_lidar = pred["box3d_lidar"].detach().cpu().numpy() # tuple
    class_names = pred["label_preds"].detach().cpu().numpy() # tuple
    # class_names_str = ['car','pedestrian','cyclist','van'] # KITTI
    class_names_str = ['car','bicycle','bus','construction_vehicle','motorcycle',
                       'pedestrian','traffic_cone','trailer','truck','barrier'] # nuscenes
    ### convert box to ros msg
    detect_object_array = DetectedObjectArray()
    detect_object_array.header = header

    print('object nums',len(class_names))

    if len(class_names) !=0:
        for i in range(len(class_names)):
            # if class_names[i] != 0:
            #     continue
            detect_object = DetectedObject()
            detect_object.header = header
            detect_object.valid = True
            detect_object.pose_reliable = True
            detect_object.velocity_reliable = False
            detect_object.acceleration_reliable = False

            detect_object.label = class_names_str[class_names[i]]

            detect_object.pose.position.x = float(boxes_lidar[i][0]) # 需要确认一下是不是0
            detect_object.pose.position.y = float(boxes_lidar[i][1]) # 需要确认一下是不是0
            detect_object.pose.position.z = float(boxes_lidar[i][2]) # 需要确认一下是不是0

            detect_object.dimensions.x = float(boxes_lidar[i][3])
            detect_object.dimensions.y = float(boxes_lidar[i][4])
            detect_object.dimensions.z = float(boxes_lidar[i][5])

            q = quaternion_from_euler(0, 0, -float(boxes_lidar[i][6]))

            detect_object.pose.orientation.x = q[0]
            detect_object.pose.orientation.y = q[1]
            detect_object.pose.orientation.z = q[2]
            detect_object.pose.orientation.w = q[3]

            detect_object_array.objects.append(detect_object)
    t3 = time.time()

    if measure_time:
        print(f" Preprocess time = {(t1-t0)* 1000:.3f} ms")
        print(f" Second network detection time = {(t2-t1)* 1000:.3f} ms")
        print(f" convert box to ros msg = {(t3-t2)* 1000:.3f} ms")

    return  detect_object_array




############### tia start ################
class SubscribePCL(object):
    def __init__(self, timer=False):
        self.points = None
        self.timer = timer
        self.header = None
    def callback_pcl(self, data):
        '''
        http://docs.ros.org/en/indigo/api/sensor_msgs/html/point__cloud2_8py_source.html
        # use pcl2.read_points to parse the data from ros topic -> python generator -> numpy array.
        # field name indicate the features we need
        Args:
            data: data from topic /velodyne_points
        Returns: none
        '''
        self.header = data.header
        t0 = time.time()
        points3D = pcl2.read_points(data, field_names=('x', 'y', 'z', 'intensity'), skip_nans=True)
        # points3D = pcl2.read_points(data, skip_nans=True)
        # print('11111111111111111111',points3D)
        # for p in points3D:
        #     print(p)
            # p  # type depends on your data type, first three e
            # ntries are probably x,y,z

        self.points = np.array([point for point in points3D])
        self.points[:, 3] = self.points[:, 3] / 255.0
        t1 = time.time()
        if self.timer:
            print(f" callback_pcl time = {(t1 - t0)* 1000:.3f} ms")



class Publish(object):
    def __init__(self,timer=False):
        self.obj_pub = rospy.Publisher('/detection/lidar_detector/objects', DetectedObjectArray,queue_size=1)
        self.timer = timer
    def result_pub(self, detect_object_array):
        t0 = time.time()

        self.obj_pub.publish(detect_object_array)
        t1 = time.time()

        if self.timer:
            print(f" pub obj time = {(t1 - t0)* 1000:.3f} ms")


if __name__ == '__main__':

    measure_time = True
    t0 = time.time()

    # Read Config file

    # config_path = "/home/ogailab/autoware.ai.10.0/src/autoware/core_perception/lidar_second/src/second/configs/all_70m/all.fhd.config"
    # config_path = "/home/ogailab/autoware.ai.10.0/src/autoware/core_perception/lidar_second/src/second/configs/nuscenes/all.fhd-puris_10epo.config"
    config_path = "/home/ogailab/autoware.ai.10.0/src/autoware/core_perception/lidar_second/src/second/configs/nuscenes/all.fhd-puris_10epo_highscores.config"

    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    # input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    # config_tool.change_detection_range(model_cfg, [-50, -50, 50, 50]) # 显存不够
    config_tool.change_detection_range(model_cfg, [-70, -32, 70, 32])  # 可以运行
    # config_tool.change_detection_range(model_cfg, [x, y, x, y])  # 可以运行
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    t1 = time.time()

    # Build Network, Target Assigner and Voxel Generator
    # ckpt_path = "/home/ogailab/autoware.ai.10.0/src/autoware/core_perception/lidar_second/70m_cb_all/voxelnet-148480.tckpt"
    ckpt_path = "/home/ogailab/tiatia/codes/TALite/model3.0/nuscene/all/fhd.rpnv2/voxelnet-140670.tckpt"
    net = build_network(model_cfg).to(device).eval()
    net.load_state_dict(torch.load(ckpt_path))
    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator
    t2 = time.time()

    # Generate Anchors
    grid_size = voxel_generator.grid_size
    feature_map_size = grid_size[:2] // config_tool.get_downsample_factor(model_cfg)
    feature_map_size = [*feature_map_size, 1][::-1]

    anchors = target_assigner.generate_anchors(feature_map_size)["anchors"]
    anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
    anchors = anchors.view(1, -1, 7)
    t3 = time.time()

    if measure_time:
        print(f" Read Config file time = {(t1-t0)* 1000:.3f} ms")
        print(f" Build Network, Target Assigner and Voxel Generator time = {(t2-t1)* 1000:.3f} ms")
        print(f" Generate Anchors time = {(t3-t2)* 1000:.3f} ms")

    rospy.init_node('SECOND_network_pub_example')
    subpcl = SubscribePCL(timer=measure_time)

    rate = rospy.Rate(10)  # 10hz

    # sub_ = rospy.Subscriber("velodyne_points", PointCloud2, subpcl.callback_pcl, queue_size=1)

    # Load Point Cloud, Generate Voxels
    pub_ = Publish(timer=measure_time)

    # rospy.spin()

    while True:
        # data = rospy.wait_for_message("/velodyne_points", PointCloud2, timeout=None)
        data = rospy.wait_for_message("/points_raw", PointCloud2, timeout=None)
        subpcl.callback_pcl(data)
        print('------------------------------------------------------frame',subpcl.points.shape)

        detect_object_array = second_inference(subpcl.header, subpcl.points, measure_time)
        #
        pub_.result_pub( detect_object_array)
        rate.sleep()





