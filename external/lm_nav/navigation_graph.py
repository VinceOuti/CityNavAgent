import pickle
from tkinter import W
import os
from PIL import Image  # type: ignore
import numpy as np
import networkx as nx  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import base64
import PIL
import io
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils_lm import latlong_to_utm

import logging


logger = logging.getLogger(__name__)


class NavigationGraph(object):
    EPS = 3e-5

    def __init__(self, path=None):

        if path is None:
            self._pos = []
            self._images = []
            self._graph = nx.Graph()
        else:
            self.load_from_file(path)

    @property
    def vert_count(self):
        return self._graph.number_of_nodes()

    def load_from_file(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        # print(data)
        self._pos = data["pos"]
        self._images = data["images"]
        self._graph = nx.readwrite.json_graph.node_link_graph(data["json_graph"])

    def add_edge(self, node, adj_node, weight=None):
        if weight is None:
            weight = np.linalg.norm(self._pos[node] - self._pos[adj_node])
        self._graph.add_edge(node, adj_node, weight=weight)

    def add_vertix(self, obs):
        assert (
           "pos" in obs
        ), "Observation should contain position"
        inx = self.vert_count
        self._graph.add_node(inx)
        self._pos = np.vstack((self._pos, obs["pos"].reshape(1, -1)))
        self._images.append(obs["image"])
        return inx

    def add_image(self, obs):
        assert (
            (b"gps/latlong" in obs) or (b"gps/utm" in obs)
        ), "Observation should contain latitude and longitude"
        if b"gps/utm" in obs:
            pos = obs[b"gps/utm"]
        else:
            pos = latlong_to_utm(obs[b"gps/latlong"])
        for inx, pos1 in enumerate(self._pos):
            if np.linalg.norm(pos - pos1) < self.EPS:
                self._images[inx].append(obs[b"images/rgb_left"])
                break
        else:
            self.add_vertix(obs)

    def cal_distance(self, pos_idx1, pos_idx2):
        # for simplicity only calculate lineardistance
        # in the future, need to modified to flying distance
        return np.linalg.norm(self._pos[pos_idx1] - self._pos[pos_idx2])

    def cal_route_length(self, route_seq):
        total_len = 0.0

        if len(route_seq) <= 1:
            print(f"route sequence len: {route_seq}. Skip route length calculation")
            return total_len

        for i in range(len(route_seq) - 1):
            total_len += self.cal_distance(route_seq[i], route_seq[i+1])

        return total_len

    def json_repr_for_visualization(self, image_size=300):
        positions = np.vstack(self._pos)
        positions[:,1] = -positions[:,1]
        min_pos = np.min(positions, 0)
        positions = positions - min_pos
        max_pos = np.max(positions)
        positions = positions / max_pos * image_size * 0.9
        positions += 0.05 * image_size
        verticies = {}
        for inx in range(self.vert_count):
            images_str = []
            for image in self._images[inx]:
                # buffered = BytesIO()
                # image = Image.fromarray(image)
                # image.save(buffered, format="JPEG")
                # img_str = str(base64.b64encode(buffered.getvalue()))[2:-1]
                img_str = str(base64.b64encode(image))[2:-1]
                images_str.append(img_str)
            verticies[str(inx)] = {
                "position": [float(z) for z in positions[inx]],
                "images": images_str,
            }

        edges = list(self._graph.edges)
        return verticies, edges

    def get_node_data(self, node_idx):
        pos = self._pos[node_idx]

        img1 = self._images[node_idx][0]
        img2 = self._images[node_idx][1]

        img1 = np.array(PIL.Image.open(io.BytesIO(img1)))
        img2 = np.array(PIL.Image.open(io.BytesIO(img2)))

        return {
            "position": pos,
            "image": [img1, img2]
        }

    def if_nearby(self, pos):
        dist, idx = self.find_closest_node(pos)
        if dist < 20.0:   # nearby threshold, consistent to the stop threshold
            return True
        else:
            return False

    # todo: update graph: cluster graph with similar location, add new edges with small distance
    def prone_graph(self):
        pass

    def find_closest_node(self, pos):
        min_dist = 1000000
        min_idx = -1
        for i in range(self.vert_count):
            node_pos = self._pos[i]
            # print(pos, node_pos)
            dist = np.linalg.norm(pos-node_pos)
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        return min_dist, min_idx
