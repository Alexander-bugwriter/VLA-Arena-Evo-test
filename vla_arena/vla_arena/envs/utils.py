import collections
import numpy as np
import os
import re
import robosuite
import xml.etree.ElementTree as ET
import time

from copy import copy
from robosuite.utils.mjcf_utils import find_elements, xml_path_completion
from robosuite.utils.placement_samplers import ObjectPositionSampler

class RandomizationError(Exception):
    """Custom exception raised when randomization fails (e.g., object placement)."""
    pass

class MultiRegionRandomSampler(ObjectPositionSampler):
    """
    Places all objects within the table uniformly random.
    Args:
        name (str): Name of this sampler.
        mujoco_objects (None or MujocoObject or list of MujocoObject): single model or list of MJCF object models
        x_range (2-array of float): Specify the (min, max) relative x_range used to uniformly place objects
        y_range (2-array of float): Specify the (min, max) relative y_range used to uniformly place objects
        rotation (None or float or Iterable):
            :`None`: Add uniform random random rotation
            :`Iterable (a,b)`: Uniformly randomize rotation angle between a and b (in radians)
            :`value`: Add fixed angle rotation
        rotation_axis (str): Can be 'x', 'y', or 'z'. Axis about which to apply the requested rotation
        ensure_object_boundary_in_range (bool):
            :`True`: The center of object is at position:
                 [uniform(min x_range + radius, max x_range - radius)], [uniform(min x_range + radius, max x_range - radius)]
            :`False`:
                [uniform(min x_range, max x_range)], [uniform(min x_range, max x_range)]
        ensure_valid_placement (bool): If True, will check for correct (valid) object placements
        reference_pos (3-array): global (x,y,z) position relative to which sampling will occur
        z_offset (float): Add a small z-offset to placements. This is useful for fixed objects
            that do not move (i.e. no free joint) to place them above the table.
    """

    def __init__(
        self,
        name,
        mujoco_objects=None,
        x_ranges=[(0, 0)],
        y_ranges=[(0, 0)],
        rotation=None,
        rotation_axis="z",
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0),
        z_offset=0.0,
    ):
        self.x_ranges = x_ranges
        self.y_ranges = y_ranges
        assert len(self.x_ranges) == len(self.y_ranges)
        self.num_ranges = len(self.x_ranges)
        self.idx = 0
        self.rotation = rotation
        self.rotation_axis = rotation_axis
        self.idx = 0

        super().__init__(
            name=name,
            mujoco_objects=mujoco_objects,
            ensure_object_boundary_in_range=ensure_object_boundary_in_range,
            ensure_valid_placement=ensure_valid_placement,
            reference_pos=reference_pos,
            z_offset=z_offset,
        )

    def _sample_x(self, object_horizontal_radius):
        """
        Samples the x location for a given object
        Args:
            object_horizontal_radius (float): Radius of the object currently being sampled for
        Returns:
            float: sampled x position
        """
        minimum, maximum = self.x_ranges[self.idx]
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return np.random.uniform(high=maximum, low=minimum)

    def _sample_y(self, object_horizontal_radius):
        """
        Samples the y location for a given object
        Args:
            object_horizontal_radius (float): Radius of the object currently being sampled for
        Returns:
            float: sampled y position
        """
        minimum, maximum = self.y_ranges[self.idx]
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return np.random.uniform(high=maximum, low=minimum)

    def _sample_quat(self):
        """
        Samples the orientation for a given object
        Returns:
            np.array: sampled (r,p,y) euler angle orientation
        Raises:
            ValueError: [Invalid rotation axis]
        """
        if self.rotation is None:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, collections.Iterable):
            rot_angle = np.random.uniform(
                high=max(self.rotation), low=min(self.rotation)
            )
        else:
            rot_angle = self.rotation

        # Return angle based on axis requested
        if self.rotation_axis == "x":
            return np.array([np.cos(rot_angle / 2), np.sin(rot_angle / 2), 0, 0])
        elif self.rotation_axis == "y":
            return np.array([np.cos(rot_angle / 2), 0, np.sin(rot_angle / 2), 0])
        elif self.rotation_axis == "z":
            return np.array([np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)])
        else:
            # Invalid axis specified, raise error
            raise ValueError(
                "Invalid rotation axis specified. Must be 'x', 'y', or 'z'. Got: {}".format(
                    self.rotation_axis
                )
            )

    def sample(self, fixtures=None, reference=None, on_top=True):
        """
        Uniformly sample relative to this sampler's reference_pos or @reference (if specified).
        Args:
            fixtures (dict): dictionary of current object placements in the scene as well as any other relevant
                obstacles that should not be in contact with newly sampled objects. Used to make sure newly
                generated placements are valid. Should be object names mapped to (pos, quat, MujocoObject)
            reference (str or 3-tuple or None): if provided, sample relative placement. Can either be a string, which
                corresponds to an existing object found in @fixtures, or a direct (x,y,z) value. If None, will sample
                relative to this sampler's `'reference_pos'` value.
            on_top (bool): if True, sample placement on top of the reference object. This corresponds to a sampled
                z-offset of the current sampled object's bottom_offset + the reference object's top_offset
                (if specified)
        Return:
            dict: dictionary of all object placements, mapping object_names to (pos, quat, obj), including the
                placements specified in @fixtures. Note quat is in (w,x,y,z) form
        Raises:
            RandomizationError: [Cannot place all objects]
            AssertionError: [Reference object name does not exist, invalid inputs]
        """
        # Standardize inputs
        placed_objects = {} if fixtures is None else copy(fixtures)
        if reference is None:
            base_offset = self.reference_pos
        elif type(reference) is str:
            assert (
                reference in placed_objects
            ), "Invalid reference received. Current options are: {}, requested: {}".format(
                placed_objects.keys(), reference
            )
            ref_pos, _, ref_obj = placed_objects[reference]
            base_offset = np.array(ref_pos)
            if on_top:
                base_offset += np.array((0, 0, ref_obj.top_offset[-1]))
        else:
            base_offset = np.array(reference)
            assert (
                base_offset.shape[0] == 3
            ), "Invalid reference received. Should be (x,y,z) 3-tuple, but got: {}".format(
                base_offset
            )

        # Sample pos and quat for all objects assigned to this sampler
        for obj in self.mujoco_objects:
            # First make sure the currently sampled object hasn't already been sampled
            assert (
                obj.name not in placed_objects
            ), "Object '{}' has already been sampled!".format(obj.name)

            horizontal_radius = obj.horizontal_radius
            bottom_offset = obj.bottom_offset
            success = False
            for i in range(5000):  # 5000 retries
                self.idx = np.random.randint(self.num_ranges)
                object_x = self._sample_x(horizontal_radius) + base_offset[0]
                object_y = self._sample_y(horizontal_radius) + base_offset[1]
                object_z = self.z_offset + base_offset[2]
                if on_top:
                    object_z -= bottom_offset[-1]

                # objects cannot overlap
                location_valid = True
                if self.ensure_valid_placement:
                    for (x, y, z), _, other_obj in placed_objects.values():
                        if (
                            np.linalg.norm((object_x - x, object_y - y))
                            <= other_obj.horizontal_radius + horizontal_radius
                        ) and (
                            object_z - z <= other_obj.top_offset[-1] - bottom_offset[-1]
                        ):
                            location_valid = False
                            break

                if location_valid:
                    # random rotation
                    quat = self._sample_quat()

                    # multiply this quat by the object's initial rotation if it has the attribute specified
                    if hasattr(obj, "init_quat"):
                        quat = quat_multiply(quat, obj.init_quat)

                    # location is valid, put the object down
                    pos = (object_x, object_y, object_z)

                    placed_objects[obj.name] = (pos, quat, obj)
                    success = True
                    break

            if not success:
                raise RandomizationError("Cannot place all objects ):")
        # print(placed_objects)
        return placed_objects

def quat_multiply(q1, q2):
    """
    Multiply two quaternions q1 and q2.
    Quaternion format: [x, y, z, w]

    Returns:
        result quaternion: [x, y, z, w]
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return np.array([x, y, z, w])

def postprocess_model_xml(xml_str, cameras_dict={}, demo_generation=False):
    """
    This function postprocesses the model.xml collected from a MuJoCo demonstration
    in order to make sure that the STL files can be found.

    Args:
        xml_str (str): Mujoco sim demonstration XML file as string

    Returns:
        str: Post-processed xml file as string
    """

    path = os.path.split(robosuite.__file__)[0]
    path_split = path.split("/")

    # replace mesh and texture file paths
    tree = ET.fromstring(xml_str)
    root = tree
    asset = root.find("asset")
    meshes = asset.findall("mesh")
    textures = asset.findall("texture")
    all_elements = meshes + textures

    # also replace paths for libero
    libero_path = os.getcwd() + "/libero"
    libero_path_split = libero_path.split("/")

    # replace paths for vla_arena
    vla_arena_path = os.getcwd() + "/vla_arena"
    vla_arena_path_split = vla_arena_path.split("/")

    for elem in all_elements:
        old_path = elem.get("file")
        if old_path is None:
            continue
        old_path_split = old_path.split("/")
        if "robosuite" in old_path_split:
            ind = max(
                loc for loc, val in enumerate(old_path_split) if val == "robosuite"
            )  # last occurrence index
            new_path_split = path_split + old_path_split[ind + 1 :]
            new_path = "/".join(new_path_split)
            elem.set("file", new_path)
        elif "libero" in old_path_split and demo_generation:
            ind = max(
                loc for loc, val in enumerate(old_path_split) if val == "libero"
            )  # last occurrence index
            new_path_split = libero_path_split + old_path_split[ind + 1 :]
            new_path = "/".join(new_path_split)
            elem.set("file", new_path)
        elif "vla_arena" in old_path_split and demo_generation:
            ind = max(
                loc for loc, val in enumerate(old_path_split) if val == "vla_arena"
            )   # last occurrence index
            new_path_split = vla_arena_path_split + old_path_split[ind + 1 :]
            new_path = "/".join(new_path_split)
            elem.set("file", new_path)
        else:
            continue

    # cameras = root.find("worldbody").findall("camera")
    cameras = find_elements(root=tree, tags="camera", return_first=False)
    for camera in cameras:
        camera_name = camera.get("name")
        if camera_name in cameras_dict:
            camera.set("name", camera_name)
            camera.set("pos", cameras_dict[camera_name]["pos"])
            camera.set("quat", cameras_dict[camera_name]["quat"])
            camera.set("mode", "fixed")

    return ET.tostring(root, encoding="utf8").decode("utf8")


def rectangle2xyrange(rect_ranges):
    x_ranges = []
    y_ranges = []
    for rect_range in rect_ranges:
        x_ranges.append([rect_range[0], rect_range[2]])
        y_ranges.append([rect_range[1], rect_range[3]])
    return x_ranges, y_ranges


class LinearMotionGenerator:
    def __init__(self, start_pos, start_quat, direction, cycle_time, travel_dist, dt=1):
        """
        往返线性运动生成器

        Args:
            start_pos: (3,) 起始位置
            start_quat: (4,) 起始四元数 (xyzw)
            direction: (3,) 运动方向（不需要归一化）
            cycle_time: float 一个往返周期总时长 (秒)
            travel_dist: float 单程最大位移 (米)
            dt: float 每步时间间隔
        """
        self.start_pos = np.asarray(start_pos, dtype=float)
        self.start_quat = np.asarray(start_quat, dtype=float)
        self.direction = np.asarray(direction, dtype=float)
        self.direction /= np.linalg.norm(self.direction)  # 归一化
        self.cycle_time = float(cycle_time)
        self.travel_dist = float(travel_dist) 
        self.dt = float(dt)
        
        # 计算周期内步数
        self.steps_per_cycle = float(round(self.cycle_time / self.dt))
        self.half_steps = self.steps_per_cycle // 2  # 单向运动步数
        
        # 单步位移
        self.step_disp = self.direction * (self.travel_dist / self.half_steps)
        
        # 当前状态
        self.pos = self.start_pos.copy()
        self.quat = self.start_quat.copy()
        
        # 当前方向标志（1: 正向，-1: 反向）
        self.forward = True
        self.step_count = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        """
        迭代一次，返回 (pos, quat)
        """
        # 更新步数
        self.step_count += 1
        
        # 检查是否需要反向
        if self.step_count > self.half_steps:
            self.step_count = 1
            self.forward = not self.forward
        
        # 更新位置
        if self.forward:
            self.pos += self.step_disp
        else:
            self.pos -= self.step_disp
        
        return self.pos.copy(), self.quat.copy()
    
    def reset(self):
        """
        重置到起点
        """
        self.pos = self.start_pos.copy()
        self.quat = self.start_quat.copy()
        self.forward = True
        self.step_count = 0


class CircularMotionGenerator:
    def __init__(self, start_pos, center_pos, start_quat, period, dt=1, normal=None):
        """
        圆周运动生成器
        
        Args:
            start_pos: (3,) 圆上的起始点
            center_pos: (3,) 圆心
            start_quat: (4,) 起始四元数 (xyzw)
            period: float 转一圈所需时间 (秒)
            dt: float 时间步长 (秒)
            normal: (3,) 圆平面法向量（默认从Z轴推断）
        """
        self.center = np.asarray(center_pos, dtype=float)
        self.start_pos = np.asarray(start_pos, dtype=float)
        self.start_quat = np.asarray(start_quat, dtype=float)
        self.period = float(period)
        self.dt = float(dt)
        
        # 圆半径向量
        radius_vec = self.start_pos - self.center
        self.radius = np.linalg.norm(radius_vec)
        if self.radius < 1e-8:
            raise ValueError("start_pos 不能和 center_pos 重合！")
        self.axis_x = radius_vec / self.radius
        
        # 角速度 (rad/s)
        self.omega = 2 * np.pi / self.period
        
        # 角度步长（弧度）
        self.angle_step = self.omega * self.dt
        
        # 圆平面法向量（如果没给就默认Z轴）
        if normal is None:
            normal = np.array([0, 0, 1.0])
        self.normal = normal / np.linalg.norm(normal)
        
        # 圆平面上的第二个轴
        self.axis_y = np.cross(self.normal, self.axis_x)
        self.axis_y /= np.linalg.norm(self.axis_y)
        
        # 当前角度
        self.angle = 0.0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """
        根据固定时间步长计算角度变化，返回 (pos, quat)
        """
        # 更新角度
        self.angle += self.angle_step
        
        # 圆周坐标
        pos = (self.center 
               + self.radius * np.cos(self.angle) * self.axis_x
               + self.radius * np.sin(self.angle) * self.axis_y)
        
        return pos.copy(), self.start_quat.copy()
    
    def reset(self):        
        self.angle = 0.0

def direction_to_quaternion(target_dir):
    """
    以(0,-1,0)对应四元数(1,1,1,1)为基准，将方向向量转换为四元数
    
    参数：
        target_dir: 目标方向向量，格式为(x, y, z)
        
    返回：
        四元数，格式为(w, x, y, z)
    """
    # 标准化目标方向向量
    target = np.array(target_dir, dtype=np.float64)
    target_norm = np.linalg.norm(target)
    
    if target_norm < 1e-6:
        raise ValueError("目标方向向量不能为零向量")
    
    target = target / target_norm
    
    # 基准参考：(0,1,0)方向对应四元数(1,1,1,1)
    # 该四元数的模长为2，我们将保持这一缩放比例
    base_quat = np.array([1.0, 1.0, 1.0, 1.0])
    base_dir = np.array([0.0, 1.0, 0.0])  # 基准方向
    ref_dir = np.array([0.0, 0.1])  # 原始参考方向(Z轴正向)
    
    # 计算从基准方向到目标方向的旋转
    # 1. 计算旋转轴（基准方向与目标方向的叉积）
    axis = np.cross(base_dir, target)
    axis_norm = np.linalg.norm(axis)
    
    # 2. 处理共线情况
    if axis_norm < 1e-6:
        # 与基准方向相同或相反
        dot = np.dot(base_dir, target)
        if dot > 0:
            # 与基准方向相同，返回基准四元数
            return tuple(base_quat)
        else:
            # 与基准方向相反，计算相反四元数
            return (-base_quat[0], -base_quat[1], -base_quat[2], -base_quat[3])
    
    # 归一化旋转轴
    axis = axis / axis_norm
    
    # 3. 计算旋转角度
    dot_product = np.dot(base_dir, target)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    # 4. 计算从基准方向到目标方向的增量四元数（单位四元数）
    half_angle = angle / 2.0
    delta_quat = np.array([
        np.cos(half_angle),
        axis[0] * np.sin(half_angle),
        axis[1] * np.sin(half_angle),
        axis[2] * np.sin(half_angle)
    ])
    
    # 5. 将基准四元数归一化后与增量四元数相乘
    base_unit = base_quat / np.linalg.norm(base_quat)  # 基准单位四元数
    composite = quaternion_multiply(delta_quat, base_unit)  # 复合旋转
    
    # 6. 缩放回原比例（保持模长为2）
    result = composite * 2.0
    
    return tuple(result)

def quaternion_multiply(q1, q2):
    """四元数乘法：q1 * q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

class SmoothWaypointMotionGenerator:
    def __init__(self, waypoints, start_quat, segment_time=10, dt=0.01, loop=False):   
        """
        平滑轨迹生成器（支持位置 & 姿态插值）

        Args:
            waypoints: [(pos, dir), ...]，pos为位置向量，dir为方向向量
            segment_time: 每段轨迹所需时间 (秒)
            dt: 每步时间间隔 (秒)
            loop: 是否循环轨迹
        """
        if len(waypoints) < 2:
            raise ValueError("waypoints 至少要有两个！")

        self.waypoints = waypoints  # 保留完整航点信息
        self.positions = [list(map(float,wp[0])) for wp in waypoints]  # 提取位置信息
        self.quaternions = [direction_to_quaternion(list(map(float,wp[1]))) for wp in waypoints]  # 提取姿态信息
        
        self.n_segments = len(waypoints) - 1
        
        self.segment_time = float(segment_time)
        self.dt = float(dt)
        self.steps_per_segment = max(1, int(self.segment_time / self.dt))  # 改为整数步
        self.loop = bool(loop)  

        # 当前段索引 & 当前步
        self.direction=1   # 当前方向
        self.seg_idx = 0
        self.step_idx = 0
        self.current_quat = start_quat.copy()  # 当前姿态

    def __iter__(self):
        return self

    def __next__(self):
        # 检查是否所有段都已完成且不循环
        if not self.loop and self.seg_idx == 0 and self.direction == -1 and self.step_idx >= self.steps_per_segment:
            raise StopIteration

        # 获取当前段的起始和结束航点
        p0 = np.array(self.positions[self.seg_idx])
        p1 = np.array(self.positions[(self.seg_idx + self.direction) % len(self.positions)])
        
        # 插值比例 alpha (0到1之间)
        alpha = self.step_idx / self.steps_per_segment
        
        # 位置线性插值 (Lerp)
        current_pos = p0 + alpha * (p1 - p0)
        
        # 姿态插值可以在这里实现，例如球面线性插值(Slerp)
        q0 = self.quaternions[self.seg_idx]
        q1 = self.quaternions[(self.seg_idx + self.direction) % len(self.quaternions)]
        norm0 = np.linalg.norm(q0)
        norm1 = np.linalg.norm(q1)
        q0 = q0 / norm0 if norm0 != 0 else [1.0, 0.0, 0.0, 0.0]
        q1 = q1 / norm1 if norm1 != 0 else [1.0, 0.0, 0.0, 0.0]
        
        # 2. 计算点积
        dot_product = q0[0]*q1[0] + q0[1]*q1[1] + q0[2]*q1[2] + q0[3]*q1[3]
        
        # 3. 确保取最短路径（点积为负则取反）
        if dot_product < 0.0:
            q1 = [-x for x in q1]
            dot_product = -dot_product
        
        # 4. 处理数值稳定性（接近时用线性插值）
        if dot_product > 0.9995:
            # 线性插值
            interpolated = [
                q0[0] + alpha * (q1[0] - q0[0]),
                q0[1] + alpha * (q1[1] - q0[1]),
                q0[2] + alpha * (q1[2] - q0[2]),
                q0[3] + alpha * (q1[3] - q0[3])
            ]
            # 重新单位化
            norm = np.linalg.norm(interpolated)
            self.current_quat = interpolated / norm if norm != 0 else [1.0, 0.0, 0.0, 0.0]
        else:
            # 5. 计算旋转角度
            theta_0 = np.arccos(dot_product)  # 初始夹角
            theta = theta_0 * alpha           # 插值角度
            sin_theta = np.sin(theta)
            sin_theta_0 = np.sin(theta_0)
            
            # 6. 计算插值系数
            s0 = np.cos(theta) - dot_product * sin_theta / sin_theta_0
            s1 = sin_theta / sin_theta_0
            
            # 7. 计算最终插值结果
            self.current_quat = [
                s0 * q0[0] + s1 * q1[0],
                s0 * q0[1] + s1 * q1[1],
                s0 * q0[2] + s1 * q1[2],
                s0 * q0[3] + s1 * q1[3]
            ]
        
        # 前进一步
        self.step_idx += 1
        
        # 如果这一段走完，切到下一段
        if self.step_idx >= self.steps_per_segment:
            self.step_idx = 0
            self.seg_idx += self.direction  # 根据方向更新段索引
            
            # 检查是否到达边界，需要改变方向
            if self.seg_idx >= self.n_segments:
                # 到达最后一个段，改为反向
                self.direction = -1
                self.seg_idx = self.n_segments  # 保持在最后一个有效段
            elif self.seg_idx <= 0:
                # 到达第一个段，若循环则改为正向，否则结束
                if self.loop:
                    self.direction = 1
                    self.seg_idx = 0
                else:
                    self.seg_idx = 0  # 准备结束

        return current_pos.tolist(), self.current_quat.copy()

    def reset(self):
        """重置轨迹到起点"""
        self.seg_idx = 0
        self.step_idx = 0

class ParabolicMotionGenerator:
    def __init__(self, start_pos, start_quat, initial_speed, direction, dt=0.01, gravity=None):
        """
        抛物线运动生成器

        Args:
            start_pos: (3,) 起始位置
            start_quat: (4,) 起始四元数 (xyzw)
            initial_speed: float 初始速度大小 (m/s)
            direction: (3,) 速度方向（不需要归一化），不应与重力方向共线
            dt: float 时间步长 (s)
            gravity: (3,) 重力加速度向量 (m/s^2)，默认为 [0, 0, -9.81]
        """
        self.start_pos = np.array(start_pos, dtype=float)
        self.start_quat = np.array(start_quat, dtype=float)
        # 速度方向归一化
        dir_vec = np.array(direction, dtype=float)
        norm = np.linalg.norm(dir_vec)
        if norm < 1e-8:
            raise ValueError("direction 向量不能为零")
        self.initial_velocity = dir_vec / norm * float(initial_speed)
        # 重力加速度
        if gravity is None:
            gravity = np.array([0, 0, -9.81], dtype=float)
        else:
            gravity = np.array(gravity, dtype=float)
        self.gravity = gravity
        self.dt = float(dt)

        # 当前状态
        self.pos = self.start_pos.copy()
        self.quat = self.start_quat.copy()
        self.t = 0.0

    def __iter__(self):
        return self

    def __next__(self):
        # pos(t) = p0 + v0 * t + 0.5 * g * t^2
        self.t += self.dt
        self.pos = self.start_pos + self.initial_velocity * self.t + 0.5 * self.gravity * self.t ** 2
        return self.pos.copy(), self.quat.copy()

    def reset(self):
        """
        重置到初始状态
        """
        self.pos = self.start_pos.copy()
        self.quat = self.start_quat.copy()
        self.t = 0.0



def make_xml_processor(body_names, random_color):
    """
    输入 body_names 列表如 ["milk_1", "lemon_1"]
    返回一个 xml_processor(xml_string) 函数
    该函数会给 xml 里每个 body_name_main 增加 mocap + weld(如果不存在)
    并为所有包含material属性的元素添加rgba属性
    """
    def xml_processor(xml_string):
        root = ET.fromstring(xml_string)

        # 找 worldbody & equality 节点
        worldbody = root.find("worldbody")
        equality = root.find("equality")
        if equality is None:
            equality = ET.SubElement(root, "equality")

        # 遍历 body_names
        for name in body_names:
            full_body_name = f"{name}_main"              # 自动补 _main
            mocap_body_name = f"{full_body_name}_mocap"  # mocap 名字

            # 检查mocap body是否已存在
            existing_mocap = worldbody.find(f".//body[@name='{mocap_body_name}']")
            if existing_mocap is None:
                # 不存在则追加 mocap body
                mocap_body = ET.Element("body", {
                    "mocap": "true",
                    "name": mocap_body_name,
                    "pos": "0 0 0"
                })
                worldbody.append(mocap_body)
                weld = ET.Element("weld", {
                    "body1": full_body_name,
                    "body2": mocap_body_name,
                    "solimp": "0.9 0.95 0.001",
                    "solref": "0.02 1"
                })
                equality.append(weld)
            else:
                # 已存在则跳过
                continue

        # 处理所有包含material属性的元素，添加rgba属性
        # 递归遍历所有元素
        def process_elements(element):
            # 检查当前元素是否有material属性
            if 'material' in element.attrib:
                material_value = element.attrib['material']
                # 如果material中不包含"robot0"且没有rgba属性，则添加
                if "robot0" not in material_value and 'rgba' not in element.attrib:
                    # 添加默认的rgba值（白色不透明）
                    color = np.append(np.random.uniform(low=0.2, high=0.8, size=3), 1)
                    color_string = ' '.join(map(str, color))
                    element.set('rgba', color_string)
                    
            # 处理子元素
            for child in element:
                process_elements(child)
                
        if random_color:
            process_elements(root)

        return ET.tostring(root, encoding="unicode")

    return xml_processor

def extract_trailing_int(s):
    """
    提取字符串末尾的整数，如果没有则返回 None。
    """
    match = re.search(r'(\d+)$', s)
    return int(match.group(1)) if match else None