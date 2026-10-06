"""CPU checks for the Blender bridge: camera conventions, preview mesh, trajectory export."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blender import bridge  # noqa: E402
from pipeline.scene_schema import padded_frame_count  # noqa: E402


def random_tcw(rng):
    angles = rng.uniform(-np.pi, np.pi, 3)
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    tcw = np.eye(4)
    tcw[:3, :3] = rz @ ry @ rx
    tcw[:3, 3] = rng.normal(size=3)
    return tcw


class CameraConventionTest(unittest.TestCase):
    def test_round_trip(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            tcw = random_tcw(rng)
            np.testing.assert_allclose(bridge.tcw_from_blender(bridge.blender_from_tcw(tcw)), tcw, atol=1e-9)

    def test_scale_is_ignored(self):
        tcw = random_tcw(np.random.default_rng(1))
        world = bridge.blender_from_tcw(tcw) @ np.diag([2.0, 2.0, 2.0, 1.0])
        np.testing.assert_allclose(bridge.tcw_from_blender(world), tcw, atol=1e-9)

    def test_first_camera_looks_along_blender_y_with_z_up(self):
        world = bridge.blender_from_tcw(np.eye(4))
        np.testing.assert_allclose(world @ [0, 0, -1, 0], [0, 1, 0, 0], atol=1e-12)  # view direction
        np.testing.assert_allclose(world @ [0, 1, 0, 0], [0, 0, 1, 0], atol=1e-12)   # up
        np.testing.assert_allclose(world[:3, 3], 0, atol=1e-12)

    def test_point_seen_by_both_conventions(self):
        """A world point lands on the same pixel in OpenCV and in Blender's camera model."""
        rng = np.random.default_rng(2)
        for k in (np.array([[534.5, 0, 410.0], [0, 531.0, 250.0], [0, 0, 1]]),
                  np.array([[620.0, 0, 416.0], [0, 626.0, 230.0], [0, 0, 1]])):
            self.check_projection(k, rng)

    def check_projection(self, k, rng):
        """Blender's camera model, including shift and pixel aspect ratio."""
        lens = bridge.blender_lens(k)
        aspect = lens["pixel_aspect_y"] / lens["pixel_aspect_x"]
        for _ in range(10):
            tcw = random_tcw(rng)
            camera_point = np.array([*rng.uniform(-1, 1, 2), rng.uniform(1, 5)])
            world_cv = np.linalg.inv(tcw) @ [*camera_point, 1]
            u, v, w = k @ camera_point
            expected = np.array([u / w, v / w])
            local = np.linalg.inv(bridge.blender_from_tcw(tcw)) @ (bridge.CV_TO_BLENDER @ world_cv)
            scale = lens["lens"] / lens["sensor_width"]
            frame_x = 0.5 + local[0] / -local[2] * scale - lens["shift_x"]
            frame_y = 0.5 + (local[1] / -local[2] * scale - lens["shift_y"]) * bridge.WIDTH / (bridge.HEIGHT * aspect)
            pixel = np.array([frame_x * bridge.WIDTH, (1 - frame_y) * bridge.HEIGHT]) - 0.5
            np.testing.assert_allclose(pixel, expected, atol=1e-6)


class DepthMeshTest(unittest.TestCase):
    k = np.array([[20.0, 0, 16], [0, 20.0, 12], [0, 0, 1]])

    def test_flat_plane(self):
        rgb = np.full((24, 32, 3), 200, np.uint8)
        vertices, faces, colors = bridge.depth_mesh(rgb, np.full((24, 32), 2.0), self.k, np.eye(4), step=4)
        self.assertEqual(len(vertices), 8 * 6)
        self.assertEqual(len(faces), 2 * 7 * 5)
        np.testing.assert_allclose(vertices[:, 1], 2.0, atol=1e-6)  # depth is Blender +Y
        self.assertTrue((colors == 200).all())

    def test_discontinuity_and_invalid_depth_are_cut(self):
        depth = np.full((24, 32), 2.0)
        depth[:, 16:] = 10.0
        depth[:4, :4] = 0
        _, faces, _ = bridge.depth_mesh(np.zeros((24, 32, 3), np.uint8), depth, self.k, np.eye(4), step=4)
        full = 2 * 7 * 5
        self.assertEqual(len(faces), full - 2 * 5 - 1)  # quads across the edge, one corner triangle


class TrajectoryTest(unittest.TestCase):
    def make_scene(self, kind, frames):
        root = Path(tempfile.mkdtemp(prefix="inspatio-bridge-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(root))
        (root / "input").mkdir()
        (root / "scene.json").write_text(json.dumps(
            {"output_id": "x", "kind": kind, "views": frames if kind == "video" else 1,
             "frames": frames, "valid_frames": frames, "fps": 15.0}))
        return root

    def test_image_trajectory_is_padded(self):
        scene = self.make_scene("image", 141)
        rng = np.random.default_rng(3)
        tcw = [random_tcw(rng) for _ in range(50)]
        meta = bridge.write_target(scene, [bridge.blender_from_tcw(m).ravel().tolist() for m in tcw])
        written = np.loadtxt(scene / "input" / "target_tcw.txt").reshape(-1, 4, 4)
        self.assertEqual((meta["frames"], meta["valid_frames"]), (padded_frame_count(50), 50))
        self.assertEqual(len(written), padded_frame_count(50))
        np.testing.assert_allclose(written[:50], tcw, atol=1e-6)
        np.testing.assert_allclose(written[50:], np.repeat(written[49:50], len(written) - 50, 0))
        self.assertEqual(json.loads((scene / "scene.json").read_text())["valid_frames"], 50)

    def test_video_trajectory_must_cover_every_frame(self):
        scene = self.make_scene("video", 30)
        with self.assertRaisesRegex(ValueError, "30 images"):
            bridge.write_target(scene, [np.eye(4).ravel().tolist()] * 29)


if __name__ == "__main__":
    unittest.main()
