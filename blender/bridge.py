#!/usr/bin/env python3
"""Bridge between the Blender add-on and the InSpatio pipeline.

Runs inside the InSpatio Python environment (never inside Blender):

    python blender/bridge.py prepare  request.json   # scene -> Blender preview
    python blender/bridge.py generate request.json   # Blender camera -> pred.mp4

Each Blender scene lives in ``blender_scenes/<name>/``, a regular scene
directory that ``run_scene_inference.py --scene_dir`` accepts. Blender data is
written to its ``blender/`` subdirectory.

Coordinates: InSpatio uses OpenCV world-to-camera matrices (x right, y down,
z forward). Blender receives the same world rotated so that OpenCV -y is up
(+Z) and OpenCV +z, the first source camera's view direction, is +Y.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.scene_depth_io import DEPTH_ENCODING, decode_image_depth, image_depth_paths  # noqa: E402
from pipeline.scene_schema import RESOLUTION, padded_frame_count  # noqa: E402

SCENES = ROOT / "blender_scenes"
OUTPUT = ROOT / "output"
HEIGHT, WIDTH = RESOLUTION
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
DEFAULT_IMAGE_FRAMES = 81

# OpenCV world -> Blender world, and OpenCV camera axes <-> Blender camera axes.
CV_TO_BLENDER = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
FLIP_YZ = np.diag([1.0, -1.0, -1.0, 1.0])


def blender_from_tcw(tcw):
    """OpenCV world-to-camera 4x4 -> Blender camera matrix_world."""
    return CV_TO_BLENDER @ np.linalg.inv(np.asarray(tcw, dtype=np.float64)) @ FLIP_YZ


def rigid(matrix):
    """Drop scale and shear from a Blender matrix_world."""
    matrix = np.array(matrix, dtype=np.float64)
    u, _, vt = np.linalg.svd(matrix[:3, :3])
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        rotation = u @ np.diag([1.0, 1.0, -1.0]) @ vt
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = matrix[:3, 3]
    return result


def tcw_from_blender(matrix_world):
    """Blender camera matrix_world -> OpenCV world-to-camera 4x4."""
    return FLIP_YZ @ np.linalg.inv(rigid(matrix_world)) @ CV_TO_BLENDER


def blender_lens(intrinsics, width=WIDTH, height=HEIGHT, sensor_width=36.0):
    """Blender camera settings matching a 3x3 pinhole matrix.

    The pipeline puts pixel centres on integer coordinates, so pixel x spans
    [x - 0.5, x + 0.5]; Blender's frame spans [0, width]. fx != fy becomes a
    render pixel aspect ratio (Blender requires both values >= 1).
    """
    k = np.asarray(intrinsics, dtype=np.float64)
    ratio = k[0, 0] / k[1, 1]
    aspect_x, aspect_y = (1.0, ratio) if ratio >= 1 else (1 / ratio, 1.0)
    return {"lens": float(k[0, 0] * sensor_width / width), "sensor_width": sensor_width,
            "shift_x": float((width / 2 - (k[0, 2] + 0.5)) / width),
            "shift_y": float((k[1, 2] + 0.5 - height / 2) * ratio / width),
            "pixel_aspect_x": float(aspect_x), "pixel_aspect_y": float(aspect_y)}


def load_matrices(path, size):
    values = np.loadtxt(path, ndmin=2).astype(np.float64)
    return values.reshape(-1, size, size)


def save_matrices(path, matrices):
    temporary = path.with_suffix(".tmp")
    np.savetxt(temporary, np.asarray(matrices).reshape(len(matrices), -1), fmt="%.9g")
    temporary.replace(path)


def depth_mesh(rgb, depth, intrinsics, tcw, step=4, max_jump=0.1):
    """Coloured triangle mesh in Blender world coordinates from one depth map.

    Triangles spanning a depth discontinuity (relative jump above max_jump)
    are dropped so foreground and background do not get connected.
    """
    height, width = depth.shape
    xs = np.arange(0, width, step)
    ys = np.arange(0, height, step)
    u, v = np.meshgrid(xs, ys)
    d = depth[v, u].astype(np.float64)
    pixels = np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
    camera = (np.linalg.inv(intrinsics) @ pixels) * d.ravel()
    world = np.linalg.inv(tcw) @ np.vstack([camera, np.ones(u.size)])
    vertices = (CV_TO_BLENDER @ world)[:3].T
    colors = rgb[v, u].reshape(-1, 3)

    rows, cols = d.shape
    index = np.arange(rows * cols).reshape(rows, cols)
    a, b = index[:-1, :-1].ravel(), index[:-1, 1:].ravel()
    c, e = index[1:, 1:].ravel(), index[1:, :-1].ravel()
    flat = d.ravel()
    valid = np.isfinite(flat) & (flat > 0)
    faces = []
    for triangle in (np.stack([a, e, b], 1), np.stack([b, e, c], 1)):
        depths = flat[triangle]
        keep = valid[triangle].all(1)
        low = np.where(keep, depths.min(1), 1.0)
        keep &= (depths.max(1) - low) <= max_jump * low
        faces.append(triangle[keep])
    faces = np.concatenate(faces)
    used = np.unique(faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return (vertices[used].astype(np.float32), remap[faces].astype(np.int32),
            colors[used].astype(np.uint8))


def save_mesh(path, vertices, faces, colors):
    np.savez(path, vertices=vertices, faces=faces, colors=colors)


def check_name(name):
    if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
        raise ValueError(f"Nom de scène invalide (lettres, chiffres, '_', '-', '.' uniquement) : {name!r}")
    return name


def read_meta(scene):
    return json.loads((scene / "scene.json").read_text())


def write_meta(scene, meta):
    temporary = scene / "scene.json.tmp"
    temporary.write_text(json.dumps(meta, indent=2) + "\n")
    temporary.replace(scene / "scene.json")


def copy_scene(source, scene):
    """Copy a prepared scene directory, keeping files already copied from it."""
    source = source.resolve(strict=True)
    if not (source / "scene.json").is_file():
        raise FileNotFoundError(f"Pas de scene.json dans {source}")
    marker = scene / "blender" / "source_scene.txt"
    if marker.is_file() and marker.read_text().strip() != str(source):
        raise ValueError(f"La scène {scene.name} vient d'un autre dossier ; choisis un autre nom")
    if not marker.is_file():
        if scene.exists() and any(scene.iterdir()):
            raise ValueError(f"{scene} existe déjà ; choisis un autre nom")
        shutil.copytree(source, scene, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("preprocessed", "*.tmp*"))
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(source) + "\n")


def create_video_scene(video, prompt, scene):
    """Scene directory for a raw video, with a placeholder target trajectory."""
    from pipeline.scene_cache import video_metadata
    video = video.resolve(strict=True)
    marker = scene / "blender" / "source_video.txt"
    if marker.is_file() and marker.read_text().strip() != str(video):
        raise ValueError(f"La scène {scene.name} vient d'une autre vidéo ; choisis un autre nom")
    if not marker.is_file() and scene.exists() and any(scene.iterdir()):
        raise ValueError(f"{scene} existe déjà ; choisis un autre nom")
    inputs = scene / "input"
    inputs.mkdir(parents=True, exist_ok=True)
    target = inputs / "video.mp4"
    if not target.is_file():
        temporary = inputs / "video.tmp.mp4"
        if video_metadata(video)[2] == (HEIGHT, WIDTH):
            shutil.copyfile(video, temporary)
        else:
            subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(video),
                 "-vf", f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT}",
                 "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(temporary)], check=True)
        temporary.replace(target)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(video) + "\n")
    frames, fps, _ = video_metadata(target)
    if not (inputs / "prompt.txt").is_file() or prompt:
        (inputs / "prompt.txt").write_text((prompt or "").strip() + "\n")
    write_meta(scene, {"output_id": scene.name, "kind": "video", "views": frames, "frames": frames,
                       "valid_frames": frames, "fps": fps, "resolution": [HEIGHT, WIDTH],
                       "target_intrinsics": "estimated_source_frame_000000_fixed",
                       "depth_encoding": DEPTH_ENCODING})


def ensure_depth(scene):
    """Convert image depth or run DA3 on a video, exactly as the runner does."""
    import run_scene_inference as runner
    from pipeline.gpu import choose_gpu
    meta = read_meta(scene)
    trajectory = scene / "input" / "target_tcw.txt"
    if meta["kind"] == "video" and (not trajectory.is_file() or
                                    len(load_matrices(trajectory, 4)) != meta["frames"]):
        # The runner checks the trajectory before estimating depth; the real
        # one is written by "generate".
        save_matrices(trajectory, np.tile(np.eye(4), (meta["frames"], 1, 1)))
    record = runner.load_record(scene)
    gpu = None
    if record["kind"] == "video":
        runner.validate_video_inputs(record)
        if not runner.video_depth_ready(record):
            print("Estimation de la profondeur et des caméras (Depth-Anything-3)…", flush=True)
            gpu = choose_gpu("auto")
    runner.prepare_depth(record, OUTPUT, "auto", ROOT / "checkpoints" / "depth", gpu)
    runner.validate_scene(record)


def read_views(scene, meta, indices):
    """(rgb, depth) for the given source indices."""
    if meta["kind"] == "image":
        images = sorted((scene / "input").glob("view_*.png"))
        depths = image_depth_paths(scene / "depth", meta["views"])
        from PIL import Image
        for index in indices:
            with Image.open(images[index]) as image:
                rgb = np.asarray(image.convert("RGB"))
            yield index, rgb, decode_image_depth(depths[index], scene / "depth" / "metadata.txt")
        return
    import cv2
    from datasets.scene_depth_io import decode_video_depth_frame, read_range
    minimum, maximum = read_range(scene / "depth" / "metadata.txt")
    captures = [cv2.VideoCapture(str(scene / "input" / "video.mp4")),
                cv2.VideoCapture(str(scene / "depth" / "depth.mp4"))]
    try:
        for index in indices:
            frames = []
            for capture in captures:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Impossible de lire l'image {index} de {scene}")
                frames.append(frame)
            yield (index, cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB),
                   decode_video_depth_frame(frames[1], minimum, maximum))
    finally:
        for capture in captures:
            capture.release()


def preview_indices(meta, video_frames):
    if meta["kind"] == "image":
        return list(range(meta["views"]))
    count = max(1, min(int(video_frames), meta["frames"]))
    return sorted({int(round(value)) for value in np.linspace(0, meta["frames"] - 1, count)})


def prepare(request):
    name = check_name(request["name"])
    scene = SCENES / name
    if request.get("video"):
        create_video_scene(Path(request["video"]), request.get("prompt"), scene)
    elif request.get("scene_dir"):
        copy_scene(Path(request["scene_dir"]), scene)
    else:
        raise ValueError("Il faut une vidéo ou un dossier de scène")
    meta = read_meta(scene)
    if meta.get("output_id") != name:
        meta["output_id"] = name
        write_meta(scene, meta)
    ensure_depth(scene)

    intrinsics = load_matrices(scene / "depth" / "source_intrinsics.txt", 3)
    source_tcw = load_matrices(scene / "depth" / "source_tcw.txt", 4)
    blender_dir = scene / "blender"
    blender_dir.mkdir(exist_ok=True)
    for old in blender_dir.glob("mesh_*.npz"):
        old.unlink()
    meshes = []
    for index, rgb, depth in read_views(scene, meta, preview_indices(meta, request.get("video_mesh_frames", 1))):
        mesh = depth_mesh(rgb, depth, intrinsics[index], source_tcw[index],
                          step=int(request.get("mesh_step", 4)))
        filename = f"mesh_{index:06d}.npz"
        save_mesh(blender_dir / filename, *mesh)
        meshes.append({"file": f"blender/{filename}", "source_index": index})
        print(f"Aperçu 3D : vue {index} ({len(mesh[0])} points)", flush=True)

    trajectory = scene / "input" / "target_tcw.txt"
    example = load_matrices(trajectory, 4)[:meta["valid_frames"]] if trajectory.is_file() else []
    if len(example) and np.allclose(example, np.eye(4)):
        example = []  # placeholder written by ensure_depth
    frames = meta["frames"] if meta["kind"] == "video" else (
        len(example) or DEFAULT_IMAGE_FRAMES)
    prompt_path = scene / "input" / "prompt.txt"
    data = {"version": 1, "name": name, "kind": meta["kind"], "views": meta["views"],
            "frames": frames, "fps": float(meta["fps"]), "width": WIDTH, "height": HEIGHT,
            "prompt": prompt_path.read_text().strip() if prompt_path.is_file() else "",
            "camera": blender_lens(intrinsics[0]),
            "source_cameras": [blender_from_tcw(m).ravel().tolist() for m in source_tcw],
            "example_cameras": [blender_from_tcw(m).ravel().tolist() for m in example],
            "meshes": meshes}
    (blender_dir / "import.json").write_text(json.dumps(data))
    print(f"INSPATIO_IMPORT={blender_dir / 'import.json'}", flush=True)


def write_target(scene, matrices):
    """Write the Blender trajectory as target_tcw.txt and update scene.json."""
    meta = read_meta(scene)
    tcw = np.stack([tcw_from_blender(np.reshape(m, (4, 4))) for m in matrices])
    count = len(tcw)
    if meta["kind"] == "video":
        if count != meta["frames"]:
            raise ValueError(f"La vidéo a {meta['frames']} images mais la caméra en a {count} : "
                             f"règle la plage d'images de Blender sur 1 – {meta['frames']}")
    else:
        frames = padded_frame_count(count)
        tcw = np.concatenate([tcw, np.repeat(tcw[-1:], frames - count, axis=0)])
        meta.update(frames=frames, valid_frames=count)
        write_meta(scene, meta)
    save_matrices(scene / "input" / "target_tcw.txt", tcw)
    return meta


def generate(request):
    name = check_name(request["name"])
    scene = SCENES / name
    if not (scene / "scene.json").is_file():
        raise FileNotFoundError(f"Scène inconnue : {name} (lance d'abord « Préparer »)")
    if request.get("prompt", "").strip():
        (scene / "input" / "prompt.txt").write_text(request["prompt"].strip() + "\n")
    meta = write_target(scene, request["matrices"])
    print(f"Trajectoire écrite : {len(request['matrices'])} images à {meta['fps']} i/s", flush=True)
    command = [sys.executable, str(ROOT / "run_scene_inference.py"), "--scene_dir", str(scene)]
    if request.get("gpu"):
        command += ["--gpu", str(request["gpu"])]
    subprocess.run(command, cwd=ROOT, check=True)
    result = OUTPUT / name / "pred.mp4"
    if not result.is_file():
        raise FileNotFoundError(result)
    renders = scene / "renders"
    renders.mkdir(exist_ok=True)
    copy = renders / f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.mp4"
    shutil.copyfile(result, copy)
    print(f"INSPATIO_RESULT={copy}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("prepare", "generate"))
    parser.add_argument("request", type=Path, help="JSON request written by the Blender add-on")
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    try:
        (prepare if args.command == "prepare" else generate)(request)
    except subprocess.CalledProcessError as error:
        print(f"INSPATIO_ERROR=La commande a échoué (code {error.returncode}) : {' '.join(map(str, error.cmd))}",
              flush=True)
        sys.exit(1)
    except Exception as error:  # reported in Blender's panel
        print(f"INSPATIO_ERROR={type(error).__name__}: {error}", flush=True)
        raise


if __name__ == "__main__":
    main()
