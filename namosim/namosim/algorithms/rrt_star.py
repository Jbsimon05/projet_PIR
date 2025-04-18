import numpy as np
import random
import math
from dataclasses import dataclass
from typing import List, Optional

from namosim.data_models import PoseModel
from namosim.utils import utils
from namosim.world.binary_occupancy_grid import BinaryOccupancyGrid
from shapely import affinity
from shapely.geometry import Polygon

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
        neighbor_radius: float = 2.0,
    ):
        """
        Initialize RRT planner for differential drive robot
        start: (x, y, theta) initial pose
        goal: (x, y, theta) goal pose
        bounds: (x_min, x_max, y_min, y_max) workspace boundaries
        """
        self.polygon = polygon
        self.start = Node(start)
        self.goal = Node(goal)
        self.map = map
        self.max_iter = max_iter
        self.goal_tolerance = goal_tolerance
        self.neighbor_radius = neighbor_radius
        self.tree: List[Node] = [self.start]
        self.max_vel = self.map.cell_size

    def random_pose(self) -> PoseModel:
        """Generate random configuration in workspace"""
        x = random.uniform(0, self.map.width)
        y = random.uniform(0, self.map.height)
        theta = random.uniform(-180, 180)
        return (x, y, theta)

    def nearest_node(self, pose: PoseModel) -> Node:
        """Find nearest node in tree to given pose"""
        distances = [
            utils.distance_between_poses(pose, node.pose) for node in self.tree
        ]
        return self.tree[np.argmin(distances)]

    def steer(self, from_node: Node, target: PoseModel) -> Node:
        """Steer by testing ranges of linear and angular velocities towards target"""
        x0, y0, theta0 = from_node.pose
        theta0_rad = utils.normalize_angle_radians(math.radians(theta0))

        # Define ranges of linear and angular velocities
        linear_vels = np.linspace(-self.max_vel, self.max_vel, 3)
        angular_vels = np.linspace(-np.pi / 8, np.pi / 8, 5)

        # Create combinations of control inputs
        control_inputs = [(v, w) for v in linear_vels for w in angular_vels]

        best_node = from_node
        best_distance = float("inf")

        # Simulate each control input
        for v, w in control_inputs:
            if v == 0 and w == 0:
                continue

            # Calculate new pose based on velocity inputs
            if abs(w) < 1e-6:
                x_new = x0 + v * math.cos(theta0_rad)
                y_new = y0 + v * math.sin(theta0_rad)
                theta_new_rad = theta0_rad
            else: # Arc motion
                x_new = x0 + (v / w) * (math.sin(theta0_rad + w) - math.sin(theta0_rad))
                y_new = y0 - (v / w) * (math.cos(theta0_rad + w) - math.cos(theta0_rad))
                theta_new_rad = theta0_rad + w
            
            # Normalize the new angle relative to the target to avoid 180-degree flips
            theta_new_rad = utils.normalize_angle_radians(theta_new_rad)
            new_pose = (x_new, y_new, math.degrees(theta_new_rad))

            # Calculate distance to target with proper angle difference
            distance_to_target = utils.distance_between_poses(new_pose, target)
            
            # Create temporary node for collision checking
            temp_node = Node(new_pose)

            if distance_to_target < best_distance and self.collision_free(temp_node):
                best_distance = distance_to_target
                best_node = Node(new_pose, from_node)
                best_node.cost = from_node.cost + utils.distance_between_poses(
                    from_node.pose, new_pose
                )
        return best_node

    def collision_free(self, node: Node) -> bool:
        """Check if the node's pose is collision-free in the map"""
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
        """Check if node is near goal"""
        return (
            utils.distance_between_poses(node.pose, self.goal.pose)
            <= self.goal_tolerance
        )

    def get_neighbors(self, node: Node) -> List[Node]:
        """Return list of nodes in the tree within neighbor_radius of the given node"""
        return [
            n for n in self.tree
            if utils.distance_between_poses(n.pose, node.pose) <= self.neighbor_radius
        ]

    def choose_parent(self, neighbors: List[Node], new_node: Node) -> Node:
        """Choose the best parent for new_node among neighbors to minimize cost"""
        min_cost = float("inf")
        best_parent = new_node.parent
        for neighbor in neighbors:
            temp_cost = neighbor.cost + utils.distance_between_poses(neighbor.pose, new_node.pose)
            if temp_cost < min_cost and self.collision_free(Node(new_node.pose, neighbor)):
                min_cost = temp_cost
                best_parent = neighbor
        if best_parent is not None:
            new_node.parent = best_parent
            new_node.cost = min_cost
        return new_node

    def rewire(self, neighbors: List[Node], new_node: Node):
        """Try to rewire the tree to improve paths through new_node"""
        for neighbor in neighbors:
            temp_cost = new_node.cost + utils.distance_between_poses(new_node.pose, neighbor.pose)
            if temp_cost < neighbor.cost and self.collision_free(Node(neighbor.pose, new_node)):
                neighbor.parent = new_node
                neighbor.cost = temp_cost

    def plan(self) -> Optional[List[Node]]:
        """Main RRT planning algorithm"""
        for n in range(self.max_iter):
            rand_config = self.random_pose()
            if random.random() < 0.1:
                rand_config = self.goal.pose
            nearest = self.nearest_node(rand_config)
            new_node = self.steer(nearest, rand_config)
            if self.collision_free(new_node):
                neighbors = self.get_neighbors(new_node)
                new_node = self.choose_parent(neighbors, new_node)
                self.tree.append(new_node)
                self.rewire(neighbors, new_node)
                if self.near_goal(new_node) and n > 5000:
                    path = self._get_path(new_node)
                    return path
        return None

    def _get_path(self, node: Node) -> List[Node]:
        """Extract path from goal to start"""
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        return path[::-1]
