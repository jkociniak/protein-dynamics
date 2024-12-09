import torch
import numpy as np
import trimesh

from src.datasets.euclidean.base import EuclideanEmbeddedSubmanifoldDataset
from src.datasets.euclidean import DatasetRegistry

REGISTRY = DatasetRegistry
register = REGISTRY.register()


@register()
class Helix3DDataset(EuclideanEmbeddedSubmanifoldDataset):
    def __init__(self, starting_angle=0., ending_angle=torch.pi, **kwargs):
        self.starting_angle = starting_angle
        self.ending_angle = ending_angle
        super().__init__(**kwargs)

    def generate_points(self):
        t = torch.linspace(self.starting_angle, self.ending_angle, self.n_points)
        x = torch.cos(t)
        y = torch.sin(t)
        z = t
        return torch.stack([x, y, z], dim=1)


@register()
class OscillatingHelix3DDataset(EuclideanEmbeddedSubmanifoldDataset):
    def __init__(self, starting_angle=0., ending_angle=torch.pi, **kwargs):
        self.starting_angle = starting_angle
        self.ending_angle = ending_angle
        super().__init__(**kwargs)

    def generate_points(self):
        t = torch.linspace(self.starting_angle, self.ending_angle, self.n_points)
        r = 1 + 0.1 * torch.sin(10 * t)
        x = r * torch.cos(t)
        y = r * torch.sin(t)
        z = t
        return torch.stack([x, y, z], dim=1)


@register()
class Sphere3DDataset(EuclideanEmbeddedSubmanifoldDataset):
    def __init__(self, polar_angle_range, azimuth_range, **kwargs):
        assert len(polar_angle_range) == 2, 'Polar angle range must be a 2-tuple'
        assert 0 <= polar_angle_range[0] < torch.pi, 'Invalid polar angle starting value'
        assert 0 <= polar_angle_range[1] < torch.pi, 'Invalid polar angle ending value'
        assert polar_angle_range[0] < polar_angle_range[1], 'Invalid polar angle range'

        assert len(azimuth_range) == 2, 'Azimuth range must be a 2-tuple'
        assert -torch.pi <= azimuth_range[0] < torch.pi, 'Invalid azimuth starting value'
        assert -torch.pi <= azimuth_range[1] < torch.pi, 'Invalid azimuth ending value'
        assert azimuth_range[0] < azimuth_range[1], 'Invalid azimuth range'

        self.polar_angle_range = polar_angle_range
        self.azimuth_range = azimuth_range
        super().__init__(**kwargs)

    def generate_points(self):
        all_points = []
        while len(all_points) < self.n_points:
            # 1. sample a standard gaussian distribution for each coordinate
            points = torch.randn(self.n_points, 3)
            # 2. normalize the points to lie on the unit sphere
            points /= points.norm(dim=1, keepdim=True)
            # 3. filter the points that are within the specified polar angle and azimuth range
            polar_angle = torch.acos(points[:, 2])
            azimuth = torch.atan2(points[:, 1], points[:, 0])
            mask = (self.polar_angle_range[0] <= polar_angle) & (polar_angle <= self.polar_angle_range[1]) & \
                   (self.azimuth_range[0] <= azimuth) & (azimuth <= self.azimuth_range[1])

            points = points[mask]
            all_points.append(points)

        points = torch.cat(all_points, dim=0)
        if points.shape[0] > self.n_points:
            points = points[:self.n_points, :]

        return points


@register()
class Torus3DDataset(EuclideanEmbeddedSubmanifoldDataset):
    def __init__(self, R=0.3, r=0.05, **kwargs):
        self.R = R
        self.r = r
        super().__init__(**kwargs)

    def generate_points(self):
        # Generate random angles
        theta = np.random.uniform(0, 2 * np.pi, self.n_points)  # angle around tube
        phi = np.random.uniform(0, 2 * np.pi, self.n_points)  # angle around torus

        # Calculate coordinates
        x = (self.R + self.r * np.cos(theta)) * np.cos(phi)
        y = (self.R + self.r * np.cos(theta)) * np.sin(phi)
        z = self.r * np.sin(theta)

        # Stack coordinates into a 100x3 matrix
        points = torch.from_numpy(np.column_stack((x, y, z))).float()
        return points


@register()
class Custom3DDataset(EuclideanEmbeddedSubmanifoldDataset):
    def __init__(self, **kwargs):
        self.line_points = None
        self.ball_2d_points = None
        self.ball_3d_points = None
        super().__init__(n_points=1120, **kwargs)

    def generate_points(self):
        # Sample points
        self.line_points = self.sample_line(20)
        self.ball_2d_points = self.sample_2d_ball(100, (2, 0, 0))
        self.ball_3d_points = self.sample_3d_ball(1000, (0, 0, 0))

        # Combine all points
        all_points = np.vstack((self.line_points, self.ball_2d_points, self.ball_3d_points))
        return torch.from_numpy(all_points).float()

    @staticmethod
    def sample_line(num_points):
        x = np.random.uniform(3, 5, num_points)  # Sampling from x > 3 to x = 5
        y = np.zeros(num_points)
        z = np.zeros(num_points)
        return np.column_stack((x, y, z))

    @staticmethod
    def sample_2d_ball(num_points, center):
        r = np.sqrt(np.random.uniform(0, 1, num_points))
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        x = r * np.cos(theta) + center[0]
        y = r * np.sin(theta) + center[1]
        z = np.zeros(num_points)
        return np.column_stack((x, y, z))

    @staticmethod
    def sample_3d_ball(num_points, center):
        r = np.cbrt(np.random.uniform(0, 1, num_points))
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        phi = np.arccos(2 * np.random.uniform(0, 1, num_points) - 1)
        x = r * np.sin(phi) * np.cos(theta) + center[0]
        y = r * np.sin(phi) * np.sin(theta) + center[1]
        z = r * np.cos(phi) + center[2]
        return np.column_stack((x, y, z))

    def plot(self, ax):
        line_points = self.line_points
        ball_2d_points = self.ball_2d_points
        ball_3d_points = self.ball_3d_points
        all_points = self.points

        # Plot each set of points with different colors
        ax.scatter(line_points[:, 0], line_points[:, 1], line_points[:, 2], c='r', label='Line')
        ax.scatter(ball_2d_points[:, 0], ball_2d_points[:, 1], ball_2d_points[:, 2], c='g', label='2D Ball')
        ax.scatter(ball_3d_points[:, 0], ball_3d_points[:, 1], ball_3d_points[:, 2], c='b', label='3D Ball')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('Custom 3D Dataset Visualization')

        # Set equal aspect ratio
        max_range = np.array([all_points[:, 0].max() - all_points[:, 0].min(),
                              all_points[:, 1].max() - all_points[:, 1].min(),
                              all_points[:, 2].max() - all_points[:, 2].min()]).max() / 2.0
        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)


@register()
class StanfordBunny3DDataset(EuclideanEmbeddedSubmanifoldDataset):
    def __init__(self, **kwargs):
        self.path = '/Users/janko/Code/Normal-vectors/data/stanford_bunny/bunny.obj'
        super().__init__(**kwargs)

    def generate_points(self):
        mesh = trimesh.load(self.path)
        return torch.from_numpy(mesh.vertices).float()
