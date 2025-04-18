import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional
import random
import math

from namosim.data_models import PoseModel
from namosim.utils import utils
from namosim.world.binary_occupancy_grid import BinaryOccupancyGrid
from shapely import affinity
from shapely.geometry import Polygon

try:
    from scipy.spatial import KDTree
    _has_kdtree = True
except ImportError:
    _has_kdtree = False

@dataclass
class Node:
    pose: PoseModel
    parent: Optional["Node"] = None
    cost: float = 0.0

class DiffDriveRRTStar:
    def __init__(
        self,
        polygon: Polygon,
        start: PoseModel,
        goal: PoseModel,
        map: BinaryOccupancyGrid,
        max_iter: int = 10000,
        goal_tolerance=0.1,
        use_kdtree: bool = True,
        informed: bool = True
    ):
        self.polygon = polygon
        self.start = Node(start)
        self.goal = Node(goal)
        self.map = map
        self.max_iter = max_iter
        self.goal_tolerance = goal_tolerance
        self.tree: List[Node] = [self.start]
        self.use_kdtree = use_kdtree and _has_kdtree
        self._kdtree = None
        self._node_coords = [self._pose_to_xy(self.start.pose)]

        self.max_vel = self.map.cell_size

        self.search_radius = self.map.cell_size * 5

        self.informed = informed
        self.best_cost = float("inf")
        self.c_best = None
        self.c_min = utils.distance_between_poses(self.start.pose, self.goal.pose)
        self.x_center = np.array([(self.start.pose[0] + self.goal.pose[0]) / 2,
                                  (self.start.pose[1] + self.goal.pose[1]) / 2])
        dx = (self.goal.pose[0] - self.start.pose[0]) / self.c_min if self.c_min > 0 else 1
        dy = (self.goal.pose[1] - self.start.pose[1]) / self.c_min if self.c_min > 0 else 0
        self.C = np.array([[dx, -dy], [dy, dx]])


    def _pose_to_xy(self, pose: PoseModel):
        return (pose[0], pose[1])

    def _update_kdtree(self):
        if self.use_kdtree:
            self._node_coords = [self._pose_to_xy(node.pose) for node in self.tree]
            self._kdtree = KDTree(self._node_coords)

    def _sample_unit_ball(self):
        # Uniform sampling in unit circle
        a = random.random()
        b = random.random()
        r = a ** 0.5
        theta = 2 * math.pi * b
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        return np.array([x, y])

    def random_pose(self) -> PoseModel:
        if self.informed and self.c_best is not None and self.c_best < float("inf"):
            # Informed sampling in an ellipse
            c_best = self.c_best
            c_min = self.c_min
            if c_best == float("inf") or c_best < c_min:
                # fallback to uniform
                x = random.uniform(0, self.map.width)
                y = random.uniform(0, self.map.height)
                theta = random.uniform(-180, 180)
                return (x, y, theta)
            else:
                # Ellipse axes
                a = c_best / 2.0
                b = math.sqrt(c_best ** 2 - c_min ** 2) / 2.0 if c_best > c_min else 0.001
                while True:
                    sample = self._sample_unit_ball()
                    # Scale and rotate
                    point = np.dot(self.C, np.array([a * sample[0], b * sample[1]]))
                    x = point[0] + self.x_center[0]
                    y = point[1] + self.x_center[1]
                    # Check bounds
                    if 0 <= x <= self.map.width and 0 <= y <= self.map.height:
                        theta = random.uniform(-180, 180)
                        return (x, y, theta)
        else:
            x = random.uniform(0, self.map.width)
            y = random.uniform(0, self.map.height)
            theta = random.uniform(-180, 180)
            return (x, y, theta)

    def nearest_node(self, pose: PoseModel) -> Node:
        if self.use_kdtree and self._kdtree is not None:
            xy = self._pose_to_xy(pose)
            _, idx = self._kdtree.query(xy)
            return self.tree[idx]
        else:
            distances = [utils.distance_between_poses(pose, node.pose) for node in self.tree]
            return self.tree[np.argmin(distances)]

    def steer(self, from_node: Node, target: PoseModel) -> Node:
        x0, y0, theta0 = from_node.pose
        theta0_rad = utils.normalize_angle_radians(math.radians(theta0))
        linear_vels = np.linspace(-self.max_vel*0.5, self.max_vel, 3)
        angular_vels = np.linspace(-np.pi / 8, np.pi / 8, 5)
        control_inputs = [(v, w) for v in linear_vels for w in angular_vels]
        best_node = from_node
        best_distance = float("inf")
        
        for v, w in control_inputs:
            if v == 0 and w == 0:
                continue
            if abs(w) < 1e-6:
                x_new = x0 + v * math.cos(theta0_rad)
                y_new = y0 + v * math.sin(theta0_rad)
                theta_new_rad = theta0_rad
            else:
                x_new = x0 + (v / w) * (math.sin(theta0_rad + w) - math.sin(theta0_rad))
                y_new = y0 - (v / w) * (math.cos(theta0_rad + w) - math.cos(theta0_rad))
                theta_new_rad = theta0_rad + w
            theta_new_rad = utils.normalize_angle_radians(theta_new_rad)
            new_pose = (x_new, y_new, math.degrees(theta_new_rad))
            distance_to_target = utils.distance_between_poses(new_pose, target)
            temp_node = Node(new_pose)
            if distance_to_target < best_distance and self.collision_free(temp_node):
                best_distance = distance_to_target
                best_node = Node(new_pose, from_node)
                # Ajoute un coût si v < 0 (marche arrière)
                best_node.cost  = from_node.cost + utils.distance_between_poses(from_node.pose, new_pose)

        return best_node

    def collision_free(self, node: Node) -> bool:
        dx, dy, dtheta = (
            node.pose[0] - self.start.pose[0],
            node.pose[1] - self.start.pose[1],
            node.pose[2] - self.start.pose[2],
        )
        new_polygon = affinity.translate(self.polygon, xoff=dx, yoff=dy)
        new_polygon = affinity.rotate(new_polygon, angle=dtheta)
        cell = self.map.pose_to_cell(node.pose[0], node.pose[1])
        occupied = self.map.grid[cell[0]][cell[1]]
        return occupied == 0

    def near_goal(self, node: Node) -> bool:
        return utils.distance_between_poses(node.pose, self.goal.pose) <= self.goal_tolerance

    def _get_path(self, node: Node) -> List[Node]:
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        return path[::-1]

    def get_near_nodes(self, node: Node) -> List[Node]:
        
        # Vectorized version for speed
        poses = np.array([n.pose for n in self.tree])
        node_pose = np.array(node.pose)

        # Compute only (x, y) distance, ignore theta for neighborhood
        diff = poses[:, :2] - node_pose[:2]
        dists = np.linalg.norm(diff, axis=1)
        mask = dists < self.search_radius

        # Exclude the node itself if present
        near_nodes = [self.tree[i] for i, m in enumerate(mask) if m and self.tree[i] is not node]
        return near_nodes

    def plan(self) -> Optional[List[Node]]:
        for n in range(self.max_iter):
            rand_config = self.random_pose()
            if random.random() < 0.1:
                rand_config = self.goal.pose
            nearest = self.nearest_node(rand_config)
            new_node = self.steer(nearest, rand_config)
            if not self.collision_free(new_node):
                continue
            near_nodes = self.get_near_nodes(new_node)
            best_parent = nearest
            best_cost = nearest.cost + utils.distance_between_poses(nearest.pose, new_node.pose)
            for near in near_nodes:
                potential_cost = near.cost + utils.distance_between_poses(near.pose, new_node.pose)
                if potential_cost < best_cost and self.collision_free(Node(new_node.pose, near)):
                    best_parent = near
                    best_cost = potential_cost
            new_node.parent = best_parent
            new_node.cost = best_cost
            self.tree.append(new_node)
            if self.use_kdtree:
                self._update_kdtree()
            for near in near_nodes:
                potential_cost = new_node.cost + utils.distance_between_poses(new_node.pose, near.pose)
                if potential_cost < near.cost and self.collision_free(Node(near.pose, new_node)):
                    near.parent = new_node
                    near.cost = potential_cost
            if self.near_goal(new_node): # and n > 2000
                path = self._get_path(new_node)
                total_cost = path[-1].cost
                if self.informed and total_cost < self.best_cost:
                    self.best_cost = total_cost
                    self.c_best = total_cost
                return path
        return None

    def smooth_path(self, path: List[Node], max_trials: int = 100) -> List[Node]:
        """Shortcutting: lisse le chemin en supprimant les points inutiles si possible."""
        if not path or len(path) < 3:
            return path
        path = path.copy()
        for _ in range(max_trials):
            if len(path) < 3:
                break
            i = random.randint(0, len(path) - 3)
            j = random.randint(i + 2, len(path) - 1)
            node_i = path[i]
            node_j = path[j]
            # Vérifie si le segment direct est sans collision
            if self._shortcut_collision_free(node_i, node_j):
                # Supprime les points intermédiaires
                path = path[:i+1] + path[j:]
        return path

    def _shortcut_collision_free(self, node_a: Node, node_b: Node, steps: int = 10) -> bool:
        """Vérifie si le segment direct entre node_a et node_b est sans collision."""
        x0, y0, t0 = node_a.pose
        x1, y1, t1 = node_b.pose
        for k in range(1, steps):
            alpha = k / steps
            x = x0 + alpha * (x1 - x0)
            y = y0 + alpha * (y1 - y0)
            theta = t0 + alpha * (t1 - t0)
            temp_node = Node((x, y, theta))
            if not self.collision_free(temp_node):
                return False
        return True

    def plot(self, path: Optional[List[Node]] = None):
        fig = plt.figure(figsize=(10, 10))
        for node in self.tree:
            if node.parent:
                plt.plot(
                    [node.pose[0], node.parent.pose[0]],
                    [node.pose[1], node.parent.pose[1]],
                    "b-",
                    alpha=0.2,
                )
        if path:
            path_x = [node.pose[0] for node in path]
            path_y = [node.pose[1] for node in path]
            plt.plot(path_x, path_y, "g-", linewidth=2)
        plt.plot(self.start.pose[0], self.start.pose[1], "bo", markersize=10)
        plt.plot(self.goal.pose[0], self.goal.pose[1], "go", markersize=10)
        plt.xlim(0, self.map.width)
        plt.ylim(0, self.map.height)
        plt.grid(True)
        plt.axis("equal")
        if self.informed:
            title = (
                f"RRT* Path Planning (informed={self.informed})\n"
                f"c_best={self.c_best if self.c_best is not None else 'None'}, c_min={self.c_min:.2f}"
            )
        else:
            title = f"RRT* Path Planning for Differential Drive Robot (use_kdtree={self.use_kdtree}, informed={self.informed})"
        plt.title(title)
        plt.show()
        plt.close(fig)
