import airsim
import numpy as np
import open3d as o3d
import networkx as nx
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R


def find_closest_node(graph, point, thresh=5, return_dist=False):
    min_distance = float('inf')
    closest_node = None
    point = np.array(point)

    for node, data in graph.nodes(data=True):
        n_pos = np.array(data['pos'])
        distance = np.linalg.norm(point-n_pos)
        if distance < min_distance:
            min_distance = distance
            closest_node = node
    # print(f"min_distance: {min_distance}")
    if min_distance < thresh:
        if return_dist:
            return closest_node, min_distance
        else:
            return closest_node
    else:
        if return_dist:
            return None, -1
        else:
            return None


def compute_shortest_path(G, A, B, weight='None'):
    node_A, _ = find_closest_node(G, A, return_dist=True)
    node_B, _ = find_closest_node(G, B, return_dist=True)
    if weight != 'None':
        node_path = nx.shortest_path(G, source=node_A, target=node_B, weight=weight)
    else:
        node_path = nx.shortest_path(G, source=node_A, target=node_B)

    path = [G.nodes[node].get('pos', None)+G.nodes[node].get('ori', None) for node in node_path]
    return path, node_path


def visualize_point_cloud(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    o3d.visualization.draw_geometries([pcd])


def statistical_filter(point_cloud, k=30, std_dev_multiplier=1.0):
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=k).fit(point_cloud)
    distances, _ = nbrs.kneighbors(point_cloud)

    mean_distances = np.mean(distances, axis=1)

    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)

    filter_idx = np.where(mean_distances < global_mean + std_dev_multiplier * global_std)[0]
    filtered_points = point_cloud[filter_idx]

    return filtered_points, filter_idx


def get_IntrinsicMatrix(fov, width, height):
    intrinsic_mat = np.zeros([3,3])
    intrinsic_mat[0, 0] = width / (2*np.tan(np.deg2rad(fov/2)))
    intrinsic_mat[1, 1] = height / (2*np.tan(np.deg2rad(fov/2)))
    intrinsic_mat[0, 2] = width / 2
    intrinsic_mat[1, 2] = height / 2
    intrinsic_mat[2, 2] = 1

    return intrinsic_mat

def get_ExtrinsicMatric(camera_pose):
    '''
    :param camera_pose: [x, y, z, rx, ry, rz, rw]
    :return:
    '''
    pos = camera_pose[:3]
    rot = camera_pose[3:]
    r1 = R.from_quat(rot).as_matrix()
    r2 = R.from_euler('x', 180, degrees=True).as_matrix()
    rotation_matrix = r1.dot(r2)

    # Create translation vector
    translation_vector = pos

    # Create extrinsic matrix
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rotation_matrix
    extrinsic_matrix[:3, 3] = translation_vector

    return extrinsic_matrix


def update_camera_pose(cur_pose, delta_yaw):
    '''
    :param cur_pose: [x, y, z, rx, ry, rz, rw]
    :param delta_yaw: yaw in radians
    :return: [x, y, z, rx, ry, rz, rw]
    '''
    cur_pos = cur_pose[:3]
    cur_rot = cur_pose[3:]
    cur_rot_euler = R.from_quat(cur_rot).as_euler('xyz')

    cur_rot_euler[2] += delta_yaw
    new_rot = np.array(list(airsim.to_quaternion(*cur_rot_euler)))
    # print(R.from_quat(new_rot).as_euler('xyz'))
    new_pose = np.concatenate([cur_pos, new_rot], axis=0)

    return new_pose


def build_semantic_point_cloud(depth_img, intrinsic_mat, bboxes, labels, BEV=False):
    height, width = depth_img.shape[:2]
    label_map = np.zeros_like(depth_img, dtype=np.int_)
    depth_map = np.zeros_like(depth_img, dtype=np.float16)
    depth_map.fill(np.inf)

    avg_depth_per_bbox = []

    def _pix_in_bbox(pix, bbox):
        px, py = pix
        x_min, y_min, x_max, y_max = bbox
        if x_min < px < x_max and y_min < py < y_max:
            return True
        else:
            return False

    # get average depth for each region
    for bbox, label in zip(bboxes, labels):
        x_min, y_min, x_max, y_max = list(map(int, bbox))
        depth_region = depth_img[x_min:x_max+1, y_min:y_max+1]
        avg_depth = depth_region.mean()
        avg_depth_per_bbox.append(avg_depth)

    # obtain label map for each pixel
    for x in range(height):
        for y in range(width):
            for i, (bbox, label) in enumerate(zip(bboxes, labels)):
                if _pix_in_bbox((x,y), bbox):
                    if avg_depth_per_bbox[i] < depth_map[x, y]:
                        depth_map[x,y] = avg_depth_per_bbox[i]
                        label_map[x,y] = label

    # create point cloud
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Normalize the pixel coordinates
    normalized_x = (u - intrinsic_mat[0, 2]) / intrinsic_mat[0, 0]
    normalized_y = (v - intrinsic_mat[1, 2]) / intrinsic_mat[1, 1]

    # Reproject to 3D space by multiplying by the depth value
    x = normalized_x * depth_img
    y = normalized_y * depth_img
    z = depth_img

    # Stack the coordinates to get the point cloud
    point_cloud = np.stack((z, -x, -y), axis=-1)

    # filter out-range points
    point_cloud[point_cloud[:, :, 2]>=50] = 0
    point_cloud[point_cloud[:, :, 2]<0] = 0

    # point_cloud[point_cloud[:, :, 0]>0] = 0
    # point_cloud[point_cloud[:, :, 1]>0] = 0
    filtered_point_cloud = point_cloud[point_cloud.any(axis=-1)]
    filtered_label_map   = label_map[point_cloud.any(axis=-1)]

    return filtered_point_cloud, filtered_label_map


def merge_point_cloud(
        base_cloud: np.ndarray,
        base_label_map: np.ndarray,
        aux_cloud: np.ndarray,
        aux_label_map: np.ndarray,
        relative_pose: np.ndarray
) -> (np.ndarray, np.ndarray):
    ext_mat = get_ExtrinsicMatric(relative_pose)
    converted_aux_cloud = aux_cloud.dot(ext_mat)
    merged_point_cloud = np.concatenate((base_cloud, converted_aux_cloud), axis=0)
    merged_label_map = np.concatenate((base_label_map, aux_label_map), axis=0)

    return merged_point_cloud, merged_label_map


def visualize_semantic_point_cloud(point_cloud_flat, label_map_flat):
    labels = np.unique(label_map_flat)
    label_num = len(labels)
    color_map = {}
    for i in range(len(labels)):
        color_map[labels[i]] = i
    # visualize point cloud
    colors = plt.cm.get_cmap('hsv', label_num)
    class_colors = [colors(i) for i in range(label_num)]
    class_colors_rgb = [(color[0], color[1], color[2]) for color in class_colors]
    # class_colors_rgb = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    class_colors = np.random.rand(len(labels), 3)

    label_colors = np.array([class_colors[color_map[label_map_flat[i]]] for i in range(len(label_map_flat))])
    # print(label_colors)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_cloud_flat)
    # point_cloud = point_cloud.voxel_down_sample(0.4)
    point_cloud.colors = o3d.utility.Vector3dVector(label_colors)  # 设置点云颜色
    # point_cloud.paint_uniform_color([0, 0, 0.8])

    o3d.visualization.draw_geometries([point_cloud])


def build_semantic_map(depth_img, fov, camera_pose, boxes, phrases, visualize=False):
    height, width = depth_img.shape[:2]
    depth_img_unorm = depth_img*100
    extrinsic_mat = get_ExtrinsicMatric(camera_pose)
    # print(extrinsic_mat)
    intrinsic_mat = get_IntrinsicMatrix(fov, width, height)

    assert len(boxes) == len(phrases)

    boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]

    class_dict = {"None": 0}

    labels = []
    for i in range(len(phrases)):
        if phrases[i] not in class_dict:
            label_idx = len(class_dict)
            class_dict[phrases[i]] = label_idx
            labels.append(label_idx)
        else:
            label_idx = class_dict[phrases[i]]
            labels.append(label_idx)

    label_num = len(class_dict)
    point_cloud, label_map = build_semantic_point_cloud(depth_img_unorm, intrinsic_mat, boxes, labels)

    point_cloud_flat = point_cloud.reshape(-1, 3)
    point_cloud_flat_homo = np.hstack((point_cloud_flat, np.ones((len(point_cloud_flat), 1))))
    point_cloud_flat_homo = point_cloud_flat_homo.dot(extrinsic_mat.T)
    point_cloud_flat = point_cloud_flat_homo[:, :3]
    label_map_flat = label_map.reshape(-1)

    if visualize:
        # visualize_semantic_point_cloud(point_cloud_flat, label_map_flat)
        visualize_point_cloud(point_cloud_flat)

    return point_cloud_flat, label_map_flat, class_dict


def build_local_point_cloud(depth_img, intrinsic_mat):
    '''
    convert depth image to point cloud in ego-centric airsim coordinate system
    Args:
        depth_image: in [height, width, 1] format
    Returns:
        point_clouds in [height, width, 3] format
    '''
    px_height, px_width = depth_img.shape[:2]
    k = intrinsic_mat

    fx = k[0, 0]
    fy = k[1, 1]
    cx = k[0, 2]
    cy = k[1, 2]

    x = np.linspace(0, px_width - 1, px_width)
    y = np.linspace(0, px_height - 1, px_height)
    xv, yv = np.meshgrid(x, y)

    Z = depth_img
    X = (xv - cx) * Z / fx
    Y = (yv - cy) * Z / fy

    # convert to ego-centric airsim coordinate system
    # point_clouds = np.stack([X, Y, Z], axis=-1)
    point_clouds = np.stack((Z, -X, -Y), axis=-1)

    return point_clouds


def build_global_point_cloud(local_pc, camera_pose):
    '''
    Args:
        camera_pose: [x, y, z, rx, ry, rz, rw]
        robot_pos: camera position in world coordinate system formatted as [X,Y, Z]
        robot_ori: camera orientation in world coordinate system formatted as [x, y, z, w]
        pc: point cloud array in camera coordinate system formatted as [height, width, 3]
    Returns:
        pc2w: point cloud array in world coordinate system formatted as [height, width, 3]
    '''
    h, w = local_pc.shape[:2]
    robot_pos = camera_pose[:3]
    robot_ori = camera_pose[3:]

    pc = local_pc.reshape(-1, 3)
    pc_homo = np.hstack((pc, np.ones((h * w, 1))))

    r1 = R.from_quat(robot_ori).as_matrix()  # extrinsic rotation in world frame coordinate system
    r2 = R.from_euler('x', 180, degrees=True).as_matrix()  # align body frame with world frame coordinate system
    # r2 = np.eye(3)
    robot_rot = r1.dot(r2)

    trans_mat = np.eye(4)
    trans_mat[:3, :3] = robot_rot
    trans_mat[:3, 3] = robot_pos

    pc2w = pc_homo.dot(trans_mat.T)  # [N, 4]
    pc2w = pc2w.reshape(h, w, 4)  # [H, W, 4]

    return pc2w[:, :, :3]  # [H, W, 3]


def convert_global_pc(depth, fov, camera_pose, mask=None):
    depth = depth.squeeze()
    depth *= 100

    height, width = depth.shape[:2]
    K = get_IntrinsicMatrix(fov=fov, height=height, width=width)

    local_pc = build_local_point_cloud(depth, K)
    global_pc = build_global_point_cloud(local_pc, camera_pose)

    if mask is not None:
        filter_idx = np.where(
            (depth > 0) & (depth < 50) & mask
        )
    else:
        filter_idx = np.where(
            (depth > 0) & (depth < 50)
        )
    return global_pc, filter_idx


def build_global_map(depth, fov, camera_pose):
    '''

    :param depth: a list of depth image
    :param fov:
    :param camera_pose: a list of camera pose corresponding to depth image
    :param mask: a list of mask
    :param phrases: a list of phrases corresponding to mask
    :return:
    '''
    all_pc = []
    all_filter_idx = []
    height, width = depth[0].shape[:2]
    K = get_IntrinsicMatrix(fov=fov, height=height, width=width)

    for i in range(len(depth)):
        d = depth[i]
        d *= 100
        d = d.squeeze()

        pose = camera_pose[i]
        filter_idx = np.where(
            (d > 0) & (d < 50)
        )

        local_pc = build_local_point_cloud(d, K)
        global_pc= build_global_point_cloud(local_pc, pose)

        all_pc.append(global_pc)
        all_filter_idx.append(filter_idx)


def visualize_nx_graph(nx_graph, node_color='black', show_label=False):
    positions_3d = nx.get_node_attributes(nx_graph, 'pos')
    positions_2d = {node: (x, y) for node, (x, y, z) in positions_3d.items()}

    plt.figure(figsize=(12, 9))
    nx.draw(nx_graph, positions_2d, with_labels=show_label, node_size=30, node_color=node_color, font_size=10)

    plt.show()


