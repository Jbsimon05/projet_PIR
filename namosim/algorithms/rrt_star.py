import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional
import random
import math
import time

from namosim.data_models import PoseModel
from namosim.utils import utils
from namosim.world.binary_occupancy_grid import BinaryOccupancyGrid
from shapely import affinity
from shapely.geometry import Polygon, Point

from namosim.algorithms.kd_tree import KDTree as CustomKDTree

def default_cost_calc(p1:PoseModel, p2:PoseModel) -> float:
    return utils.distance_between_poses(p1, p2)

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
        cost_calc = default_cost_calc,
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
        self.use_kdtree = use_kdtree
        self._kdtree = None
        self.cost_calc = cost_calc

        self.max_vel = self.map.cell_size
        self.search_radius = self.map.cell_size * 5
        self.informed = informed
        self.best_cost = float("inf")
        self.c_best = None
        self.c_min = self.cost_calc(self.start.pose, self.goal.pose)
        self.x_center = np.array([
            (self.start.pose[0] + self.goal.pose[0]) / 2,
            (self.start.pose[1] + self.goal.pose[1]) / 2
        ])
        dx = (self.goal.pose[0] - self.start.pose[0]) / self.c_min if self.c_min > 0 else 1
        dy = (self.goal.pose[1] - self.start.pose[1]) / self.c_min if self.c_min > 0 else 0
        self.C = np.array([[dx, -dy], [dy, dx]])

        # Initialize KD-tree if needed
        if self.use_kdtree:
            self._kdtree = CustomKDTree(
                dimensions=2,
                point_getter=lambda node: node.pose[:2]
            )
            self._kdtree.add(self.start)

        # Initialize placeholder for timing
        self.elapsed_time: Optional[float] = None

    def _pose_to_xy(self, pose: PoseModel):
        return (pose[0], pose[1])

    def random_pose(self) -> PoseModel:
        if self.informed and self.c_best is not None and self.c_best < float("inf"):
            c_best, c_min = self.c_best, self.c_min
            if c_best == float("inf") or c_best < c_min:
                x = random.uniform(0, self.map.width)
                y = random.uniform(0, self.map.height)
                theta = random.uniform(-180, 180)
                return (x, y, theta)
            a = c_best / 2.0
            b = math.sqrt(c_best**2 - c_min**2) / 2.0 if c_best > c_min else 0.001
            while True:
                sample = self._sample_unit_ball()
                point = np.dot(self.C, np.array([a * sample[0], b * sample[1]]))
                x = point[0] + self.x_center[0]
                y = point[1] + self.x_center[1]
                if 0 <= x <= self.map.width and 0 <= y <= self.map.height:
                    theta = random.uniform(-180, 180)
                    return (x, y, theta)
        x = random.uniform(0, self.map.width)
        y = random.uniform(0, self.map.height)
        theta = random.uniform(-180, 180)
        return (x, y, theta)

    def nearest_node(self, pose: PoseModel) -> Node:
        if self.use_kdtree and self._kdtree is not None:
            result = self._kdtree.query(pose[:2], k=1)
            if result:
                return result[0]
        distances = [self.cost_calc(pose, node.pose) for node in self.tree]
        return self.tree[int(np.argmin(distances))]

    def steer(self, from_node: Node, target: PoseModel) -> Node:
        x0, y0, theta0 = from_node.pose
        theta0_rad = utils.normalize_angle_radians(math.radians(theta0))

        linear_vels = np.linspace(-self.max_vel*0.5, self.max_vel, 3)
        angular_vels = np.linspace(-np.pi / 8, np.pi / 8, 5)
        control_inputs = [(v, w) for v in linear_vels for w in angular_vels]

        best_node = from_node
        best_distance = float('inf')
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

            distance_to_target = self.cost_calc(new_pose, target)
            temp_node = Node(new_pose)
            if distance_to_target < best_distance and self.collision_free(temp_node):
                best_distance = distance_to_target
                best_node = Node(new_pose, from_node)
                best_node.cost = from_node.cost + self.cost_calc(from_node.pose, new_pose)
        return best_node

    def collision_free(self, node: Node) -> bool:
        dx, dy, dtheta = (
            node.pose[0] - self.start.pose[0],
            node.pose[1] - self.start.pose[1],
            node.pose[2] - self.start.pose[2],
        )
        new_polygon = affinity.translate(self.polygon, xoff=dx, yoff=dy)
        new_polygon = affinity.rotate(new_polygon, angle=dtheta, origin=Point(node.pose[0], node.pose[1]))

        occupied = self.map.polygon_has_collisions(new_polygon)
        return not occupied

    def near_goal(self, node: Node) -> bool:
        return self.cost_calc(node.pose, self.goal.pose) <= self.goal_tolerance

    def get_near_nodes(self, node: Node) -> List[Node]:
        if self.use_kdtree and self._kdtree is not None:
            candidates = self._kdtree.query_radius(node.pose[:2], self.search_radius)
            return [n for n in candidates if n is not node]
        poses = np.array([n.pose for n in self.tree])
        node_pose = np.array(node.pose)
        dists = np.linalg.norm(poses[:, :2] - node_pose[:2], axis=1)
        return [self.tree[i] for i in np.where(dists < self.search_radius)[0] if self.tree[i] is not node]

    def _sample_unit_ball(self):
        # Uniform sampling in unit circle
        a = random.random()
        b = random.random()
        r = a ** 0.5
        theta = 2 * math.pi * b
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        return np.array([x, y])

    def plan(self) -> Optional[List[Node]]:
        start_time = time.time()
        best_path = None  # Ajout pour stocker le meilleur chemin
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
            best_cost = nearest.cost + self.cost_calc(nearest.pose, new_node.pose)
            for near in near_nodes:
                potential_cost = near.cost + self.cost_calc(near.pose, new_node.pose)
                if potential_cost < best_cost and self.collision_free(Node(new_node.pose, near)):
                    best_parent, best_cost = near, potential_cost
            new_node.parent = best_parent
            new_node.cost = best_cost
            self.tree.append(new_node)
            if self.use_kdtree:
                self._kdtree.add(new_node)
            for near in near_nodes:
                potential_cost = new_node.cost + self.cost_calc(new_node.pose, near.pose)
                if potential_cost < near.cost and self.collision_free(Node(near.pose, new_node)):
                    near.parent, near.cost = new_node, potential_cost
            if self.near_goal(new_node):
                path = self._get_path(new_node)
                total_cost = path[-1].cost
                if self.informed:
                    if total_cost < self.best_cost:
                        self.best_cost, self.c_best = total_cost, total_cost
                        best_path = path  # On garde le meilleur chemin trouvé
                else:
                    self.elapsed_time = time.time() - start_time
                    return path
        # No path found or fin de boucle pour informed
        self.elapsed_time = time.time() - start_time
        if self.informed and best_path is not None:
            return best_path
        return None

    def smooth_path(self, path: List[Node], max_trials: int = 100) -> List[Node]:
        if len(path) < 3:
            return path
        for _ in range(max_trials):
            if len(path) < 3:
                break
            i = random.randint(0, len(path) - 3)
            j = random.randint(i + 2, len(path) - 1)
            if self._shortcut_collision_free(path[i], path[j]):
                path = path[:i+1] + path[j:]
        return path

    def _get_path(self, node: Node) -> List[Node]:
        path, curr = [], node
        while curr:
            path.append(curr)
            curr = curr.parent
        return path[::-1]

    def _shortcut_collision_free(self, node_a: Node, node_b: Node, steps: int = 10) -> bool:
        x0, y0, t0 = node_a.pose
        x1, y1, t1 = node_b.pose
        for k in range(1, steps):
            alpha = k / steps
            x = x0 + alpha * (x1 - x0)
            y = y0 + alpha * (y1 - y0)
            theta = t0 + alpha * (t1 - t0)
            if not self.collision_free(Node((x, y, theta))):
                return False
        return True

    def plot(self, path: Optional[List[Node]] = None):
        fig = plt.figure(figsize=(10, 10))
        for node in self.tree:
            if node.parent:
                plt.plot([node.pose[0], node.parent.pose[0]], [node.pose[1], node.parent.pose[1]], 'b-', alpha=0.2)
        if path:
            xs, ys = zip(*[(n.pose[0], n.pose[1]) for n in path])
            plt.plot(xs, ys, 'g-', linewidth=2)
        plt.plot(self.start.pose[0], self.start.pose[1], 'bo', markersize=10)
        plt.plot(self.goal.pose[0], self.goal.pose[1], 'go', markersize=10)
        plt.xlim(0, self.map.width)
        plt.ylim(0, self.map.height)
        plt.grid(True)
        plt.axis('equal')
        time_info = f"{self.elapsed_time:.2f}s" if self.elapsed_time is not None else "N/A"
        title = (
            f"RRT* Path Planning (informed={self.informed})\n"
            f"c_best={self.c_best}, c_min={self.c_min:.2f}\n"
            f"Time: {time_info}"
        )
        plt.title(title)
        plt.show()
        plt.close(fig)