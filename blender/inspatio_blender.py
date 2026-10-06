"""InSpatio-World dans Blender : tu animes la caméra, le modèle génère la vidéo.

Installation : Édition > Préférences > Modules complémentaires > Installer depuis
le disque… > ce fichier. Voir LISEZMOI.md, section « Blender ».

L'add-on ne fait aucun calcul lourd : il lance ``blender/bridge.py`` avec le
Python de l'environnement InSpatio (directement sous Linux, via WSL sous
Windows) et relit ce que ce script écrit dans ``blender_scenes/<nom>/``.
"""

bl_info = {
    "name": "InSpatio-World",
    "author": "In-spatio (modèle : InSpatio Team)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "Vue 3D > Barre latérale (N) > InSpatio",
    "description": "Génère une vidéo InSpatio-World 1.5 à partir d'une caméra animée dans Blender",
    "category": "Camera",
}

import json
import os
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

ADDON_KEY = __package__ or __name__
CAMERA_NAME = "InSpatio_Camera"
SOURCE_RIG_NAME = "Trajet_source"
EXAMPLE_NAME = "Trajet_exemple"
COLLECTION_NAME = "InSpatio"
MATERIAL_NAME = "InSpatio_couleurs"
COLOR_ATTRIBUTE = "Couleur"
LOG_TEXT = "InSpatio_journal"
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
LOG_LINES = 400


# --------------------------------------------------------------------------
# Paths: "runner" paths are seen by the InSpatio Python (Linux side under
# WSL); "Blender" paths are the same files as Blender sees them.

def preferences():
    return bpy.context.preferences.addons[ADDON_KEY].preferences


def use_wsl():
    return preferences().mode == "WSL"


def to_runner(path):
    path = bpy.path.abspath(path) if path.startswith("//") else path
    if not use_wsl():
        return os.path.abspath(os.path.expanduser(path))
    slashed = path.replace("\\", "/")
    match = re.match(r"^//(?:wsl\.localhost|wsl\$)/[^/]+(/.*)?$", slashed, re.IGNORECASE)
    if match:
        return match.group(1) or "/"
    match = re.match(r"^([A-Za-z]):/*(.*)$", slashed)
    if match:
        return f"/mnt/{match.group(1).lower()}/{match.group(2)}".rstrip("/")
    return slashed


def to_blender(path):
    if not use_wsl():
        return path
    match = re.match(r"^/mnt/([a-z])/(.*)$", path)
    if match:
        return f"{match.group(1).upper()}:\\" + match.group(2).replace("/", "\\")
    return f"\\\\wsl.localhost\\{preferences().wsl_distro}" + path.replace("/", "\\")


def repo_runner():
    return to_runner(preferences().repo_path).rstrip("/") or "/"


def write_request(name, command, data):
    """Write a JSON request where the bridge can read it; return its runner path."""
    runner = f"{repo_runner()}/blender_scenes/_requests/{name}_{command}.json"
    path = Path(to_blender(runner))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return runner


def bridge_command(command, request):
    prefs = preferences()
    repo = repo_runner()
    args = [prefs.python_path.strip(), "-u", f"{repo}/blender/bridge.py", command, request]
    if use_wsl():
        args = ["wsl.exe", "-d", prefs.wsl_distro, "--cd", repo, "--exec", *args]
        return args, None
    if os.sep in args[0] or "/" in args[0]:
        args[0] = to_runner(args[0])
    return args, repo


def check_setup():
    prefs = preferences()
    if not prefs.repo_path.strip() or not prefs.python_path.strip():
        return "Renseigne le dossier In-spatio et le Python InSpatio dans les préférences de l'add-on"
    if not Path(to_blender(f"{repo_runner()}/blender/bridge.py")).is_file():
        return f"blender/bridge.py introuvable dans {prefs.repo_path}"
    if use_wsl() and not prefs.wsl_distro.strip():
        return "Indique le nom de la distribution WSL dans les préférences"
    return None


# --------------------------------------------------------------------------
# Background jobs: one bridge process at a time, polled from a Blender timer.

class Job:
    def __init__(self, kind, scene_name, args, cwd, on_success):
        self.kind = kind
        self.scene_name = scene_name
        self.on_success = on_success
        self.lines = []
        self.markers = {}
        self.queue = queue.Queue()
        env = {key: value for key, value in os.environ.items()
               if key not in ("PYTHONHOME", "PYTHONPATH")}
        env["PYTHONUNBUFFERED"] = "1"
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        self.process = subprocess.Popen(
            args, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            bufsize=1, creationflags=flags)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        for line in self.process.stdout:
            self.queue.put(line.rstrip("\n"))
        self.process.stdout.close()

    def drain(self):
        changed = False
        while True:
            try:
                line = self.queue.get_nowait()
            except queue.Empty:
                return changed
            changed = True
            match = re.match(r"^(INSPATIO_[A-Z]+)=(.*)$", line)
            if match:
                self.markers[match.group(1)] = match.group(2)
            if line.strip():
                self.lines.append(line)
        del self.lines[:-LOG_LINES]

    def finished(self):
        return self.process.poll() is not None and not self.reader.is_alive()


_job = None


def job_running():
    return _job is not None and not _job.finished()


def start_job(kind, scene, args, cwd, on_success):
    global _job
    _job = Job(kind, scene.name, args, cwd, on_success)
    scene.inspatio.status = "Préparation…" if kind == "prepare" else "Génération…"
    scene.inspatio.error = ""
    if not bpy.app.timers.is_registered(poll_job):
        bpy.app.timers.register(poll_job, first_interval=0.5)


def update_log(lines):
    text = bpy.data.texts.get(LOG_TEXT) or bpy.data.texts.new(LOG_TEXT)
    text.clear()
    text.write("\n".join(lines) + "\n")


def redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


HINTS = (("Missing Wan T5 weights", "Modèles absents : lance « bash pipeline/download.sh »"),
         ("checkpoints", "Modèles absents : lance « bash pipeline/download.sh »"),
         ("No GPU is visible", "Aucune carte NVIDIA visible (nvidia-smi)"),
         ("nvidia-smi", "Pilote NVIDIA introuvable (nvidia-smi)"),
         ("out of memory", "Mémoire de la carte graphique insuffisante"),
         ("ffmpeg", "ffmpeg manquant : sudo apt install ffmpeg"))


def failure_reason(job):
    """The child's own exception line beats the bridge's generic message."""
    errors = [line for line in job.lines if re.match(r"^[\w.]+(Error|Exception): ", line)]
    reason = (errors[-1] if errors else job.markers.get("INSPATIO_ERROR")
              or (job.lines[-1] if job.lines else f"code {job.process.returncode}"))
    for needle, hint in HINTS:
        if needle.lower() in reason.lower():
            return f"{hint} — {reason}"
    return reason


def poll_job():
    job = _job
    if job is None:
        return None
    scene = bpy.data.scenes.get(job.scene_name)
    if job.drain():
        update_log(job.lines)
        if scene and job.lines:
            scene.inspatio.status = job.lines[-1][-120:]
    if not job.finished():
        redraw()
        return 0.5
    job.drain()
    update_log(job.lines)
    if scene is not None:
        props = scene.inspatio
        if job.process.returncode == 0:
            try:
                job.on_success(scene, job)
                props.status = "Terminé"
            except Exception as error:  # shown in the panel
                props.status = "Échec"
                props.error = f"{type(error).__name__}: {error}"
        else:
            props.status = "Échec" if job.process.returncode > 0 else "Annulé"
            props.error = failure_reason(job)
    redraw()
    return None


# --------------------------------------------------------------------------
# Scene construction from blender_scenes/<name>/blender/import.json

def srgb_to_linear(values):
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def color_material():
    material = bpy.data.materials.get(MATERIAL_NAME)
    if material is not None:
        return material
    material = bpy.data.materials.new(MATERIAL_NAME)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    color = nodes.new("ShaderNodeVertexColor")
    color.layer_name = COLOR_ATTRIBUTE
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")
    color.location, emission.location, output.location = (-400, 0), (-150, 0), (100, 0)
    material.node_tree.links.new(color.outputs["Color"], emission.inputs["Color"])
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def mesh_object(name, path, collection):
    arrays = np.load(path)
    vertices, faces, colors = arrays["vertices"], arrays["faces"], arrays["colors"]
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(vertices))
    mesh.vertices.foreach_set("co", vertices.astype(np.float32).ravel())
    mesh.loops.add(faces.size)
    mesh.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
    mesh.polygons.add(len(faces))
    mesh.polygons.foreach_set("loop_start", np.arange(0, faces.size, 3, dtype=np.int32))
    mesh.update()
    attribute = mesh.color_attributes.new(COLOR_ATTRIBUTE, "FLOAT_COLOR", "POINT")
    rgba = np.ones((len(colors), 4), dtype=np.float32)
    rgba[:, :3] = srgb_to_linear(colors.astype(np.float32) / 255)
    attribute.data.foreach_set("color", rgba.ravel())
    mesh.color_attributes.active_color = attribute
    mesh.materials.append(color_material())
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.hide_select = True
    return obj, vertices


def matrix(values):
    return Matrix(np.reshape(values, (4, 4)).tolist())


def animate(obj, matrices, frame_start):
    """One key per frame; quaternions avoid Euler flips on recorded paths."""
    obj.rotation_mode = "QUATERNION"
    previous = None
    for offset, values in enumerate(matrices):
        location, rotation, _ = matrix(values).decompose()
        if previous is not None:
            rotation.make_compatible(previous)
        previous = rotation
        obj.location, obj.rotation_quaternion = location, rotation
        frame = frame_start + offset
        obj.keyframe_insert("location", frame=frame)
        obj.keyframe_insert("rotation_quaternion", frame=frame)


def apply_lens(camera_data, lens, extent):
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.sensor_width = lens["sensor_width"]
    camera_data.lens = lens["lens"]
    camera_data.shift_x = lens["shift_x"]
    camera_data.shift_y = lens["shift_y"]
    camera_data.clip_start = max(extent * 1e-4, 1e-3)
    camera_data.clip_end = max(extent * 4, 100.0)


def reference_camera(name, lens, extent, collection):
    data = bpy.data.cameras.new(name)
    apply_lens(data, lens, extent)
    data.display_size = extent * 0.03
    obj = bpy.data.objects.new(name, data)
    collection.objects.link(obj)
    obj.hide_render = True
    return obj


def clear_collection(collection, keep):
    for obj in list(collection.objects):
        if obj.name in keep:
            continue
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data is not None and data.users == 0:
            if isinstance(data, bpy.types.Mesh):
                bpy.data.meshes.remove(data)
            elif isinstance(data, bpy.types.Camera):
                bpy.data.cameras.remove(data)


def build_scene(scene, import_path):
    import_path = Path(import_path)
    data = json.loads(import_path.read_text(encoding="utf-8"))
    base = import_path.parent.parent
    props = scene.inspatio
    collection = bpy.data.collections.get(COLLECTION_NAME)
    if collection is None:
        collection = bpy.data.collections.new(COLLECTION_NAME)
    if collection.name not in scene.collection.children:
        scene.collection.children.link(collection)

    camera = bpy.data.objects.get(CAMERA_NAME)
    keep_camera = camera is not None and props.scene_name == data["name"]
    if camera is not None and not keep_camera:
        bpy.data.objects.remove(camera, do_unlink=True)
        camera = None
    clear_collection(collection, {CAMERA_NAME})

    points = []
    for item in data["meshes"]:
        _, vertices = mesh_object(f"Scene_{item['source_index']:04d}", base / item["file"], collection)
        points.append(vertices)
    points = np.concatenate(points) if points else np.zeros((1, 3))
    extent = float(np.percentile(np.linalg.norm(points, axis=1), 95)) or 1.0

    lens = data["camera"]
    frame_start = 1
    sources = data["source_cameras"]
    rig = None
    if data["kind"] == "video":
        rig = reference_camera(SOURCE_RIG_NAME, lens, extent, collection)
        animate(rig, sources, frame_start)
    else:
        for index, values in enumerate(sources):
            source = reference_camera(f"Source_vue_{index:02d}", lens, extent, collection)
            source.matrix_world = matrix(values)
    if data["example_cameras"]:
        example = reference_camera(EXAMPLE_NAME, lens, extent, collection)
        animate(example, data["example_cameras"], frame_start)
        example.hide_set(True)

    if camera is None:
        camera_data = bpy.data.cameras.new(CAMERA_NAME)
        camera = bpy.data.objects.new(CAMERA_NAME, camera_data)
        collection.objects.link(camera)
        if rig is not None:
            camera.parent = rig  # follows the original path until the user moves it
            camera.matrix_parent_inverse = Matrix.Identity(4)
            camera.matrix_basis = Matrix.Identity(4)
        else:
            camera.matrix_world = matrix(sources[0])
    elif camera.name not in collection.objects:
        collection.objects.link(camera)
    apply_lens(camera.data, lens, extent)
    camera.data.display_size = extent * 0.05

    render = scene.render
    render.resolution_x, render.resolution_y = data["width"], data["height"]
    render.resolution_percentage = 100
    render.pixel_aspect_x = lens.get("pixel_aspect_x", 1.0)
    render.pixel_aspect_y = lens.get("pixel_aspect_y", 1.0)
    fps = data["fps"]
    render.fps = max(1, round(fps))
    render.fps_base = render.fps / fps
    scene.frame_start = frame_start
    scene.frame_end = frame_start + data["frames"] - 1
    scene.frame_set(frame_start)
    scene.camera = camera

    props.scene_name = data["name"]
    props.kind = data["kind"]
    props.frames = data["frames"]
    props.lens = lens["lens"]
    if data.get("prompt") and not props.prompt.strip():
        props.prompt = data["prompt"]
    show_through_camera(extent)


def show_through_camera(extent=None):
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            space = area.spaces.active
            space.shading.type = "SOLID"
            space.shading.light = "FLAT"
            space.shading.color_type = "VERTEX"  # shown as "Attribute" in the UI
            if extent:
                space.clip_end = max(space.clip_end, extent * 4)
            space.region_3d.view_perspective = "CAMERA"


def show_result(scene, path):
    camera = bpy.data.objects.get(CAMERA_NAME)
    if camera is None:
        return
    clip = bpy.data.movieclips.load(path, check_existing=True)
    clip.frame_start = scene.frame_start
    camera.data.background_images.clear()
    background = camera.data.background_images.new()
    background.source = "MOVIE_CLIP"
    background.clip = clip
    background.alpha = 1.0
    background.display_depth = "FRONT"
    background.frame_method = "FIT"
    camera.data.show_background_images = True


def camera_matrices(scene, camera):
    current = scene.frame_current
    matrices = []
    try:
        for frame in range(scene.frame_start, scene.frame_end + 1):
            scene.frame_set(frame)
            matrices.append([value for row in camera.matrix_world for value in row])
    finally:
        scene.frame_set(current)
    return matrices


def default_name(path):
    stem = Path(path.rstrip("/\\")).stem if path else ""
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", stem).lstrip("._-")
    return name or "scene"


# --------------------------------------------------------------------------
# Operators

class INSPATIO_OT_prepare(bpy.types.Operator):
    bl_idname = "inspatio.prepare"
    bl_label = "Préparer la scène"
    bl_description = "Calcule la profondeur si besoin et importe la scène en 3D avec sa caméra"

    def execute(self, context):
        props = context.scene.inspatio
        problem = check_setup()
        if job_running():
            problem = "Une tâche InSpatio est déjà en cours"
        source = props.video_path if props.source_mode == "VIDEO" else props.scene_dir
        if not problem and not source.strip():
            problem = "Choisis une vidéo ou un dossier de scène"
        if not problem and props.source_mode == "VIDEO" and not props.prompt.strip():
            problem = "Une vidéo demande une description (prompt), de préférence en anglais"
        name = props.new_name.strip() or default_name(source)
        if not problem and not NAME_PATTERN.fullmatch(name):
            problem = "Nom de scène : lettres, chiffres, « _ », « - » et « . » uniquement"
        if problem:
            self.report({"ERROR"}, problem)
            return {"CANCELLED"}
        request = {"name": name, "mesh_step": props.mesh_step, "video_mesh_frames": props.video_mesh_frames}
        if props.source_mode == "VIDEO":
            request.update(video=to_runner(source), prompt=props.prompt)
        else:
            request["scene_dir"] = to_runner(source)
        args, cwd = bridge_command("prepare", write_request(name, "prepare", request))

        def done(scene, job):
            path = job.markers.get("INSPATIO_IMPORT")
            if not path:
                raise RuntimeError("Le script n'a pas produit de scène")
            build_scene(scene, to_blender(path))

        start_job("prepare", context.scene, args, cwd, done)
        return {"FINISHED"}


class INSPATIO_OT_generate(bpy.types.Operator):
    bl_idname = "inspatio.generate"
    bl_label = "Générer la vidéo"
    bl_description = "Envoie la trajectoire de InSpatio_Camera au modèle et lance la génération"

    def execute(self, context):
        scene = context.scene
        props = scene.inspatio
        camera = bpy.data.objects.get(CAMERA_NAME)
        count = scene.frame_end - scene.frame_start + 1
        problem = check_setup()
        if job_running():
            problem = "Une tâche InSpatio est déjà en cours"
        elif not props.scene_name or camera is None:
            problem = "Prépare d'abord une scène (étape 1)"
        elif props.kind == "video" and count != props.frames:
            problem = (f"La vidéo a {props.frames} images : règle la plage de Blender sur "
                       f"{scene.frame_start} – {scene.frame_start + props.frames - 1}")
        if problem:
            self.report({"ERROR"}, problem)
            return {"CANCELLED"}
        if abs(camera.data.lens - props.lens) > 1e-3:
            self.report({"WARNING"}, "Focale modifiée : le modèle utilise toujours la focale d'origine")
        request = {"name": props.scene_name, "prompt": props.prompt, "gpu": preferences().gpu.strip(),
                   "matrices": camera_matrices(scene, camera)}
        args, cwd = bridge_command("generate", write_request(props.scene_name, "generate", request))

        def done(scene, job):
            path = job.markers.get("INSPATIO_RESULT")
            if not path:
                raise RuntimeError("Pas de vidéo produite")
            scene.inspatio.result_path = to_blender(path)
            show_result(scene, scene.inspatio.result_path)

        start_job("generate", scene, args, cwd, done)
        return {"FINISHED"}


class INSPATIO_OT_cancel(bpy.types.Operator):
    bl_idname = "inspatio.cancel"
    bl_label = "Arrêter"
    bl_description = "Arrête la tâche en cours"

    def execute(self, context):
        if job_running():
            _job.process.terminate()
        return {"FINISHED"}


class INSPATIO_OT_view_camera(bpy.types.Operator):
    bl_idname = "inspatio.view_camera"
    bl_label = "Voir par la caméra"
    bl_description = "Sélectionne InSpatio_Camera et regarde à travers elle"

    def execute(self, context):
        camera = bpy.data.objects.get(CAMERA_NAME)
        if camera is None:
            self.report({"ERROR"}, "Pas de caméra InSpatio : prépare une scène")
            return {"CANCELLED"}
        context.scene.camera = camera
        for obj in context.selected_objects:
            obj.select_set(False)
        camera.select_set(True)
        context.view_layer.objects.active = camera
        show_through_camera()
        return {"FINISHED"}


class INSPATIO_OT_open_result(bpy.types.Operator):
    bl_idname = "inspatio.open_result"
    bl_label = "Ouvrir la vidéo"
    bl_description = "Ouvre la dernière vidéo générée dans le lecteur du système"

    def execute(self, context):
        path = context.scene.inspatio.result_path
        if not path or not Path(path).is_file():
            self.report({"ERROR"}, "Aucune vidéo générée")
            return {"CANCELLED"}
        bpy.ops.wm.path_open(filepath=path)
        return {"FINISHED"}


class INSPATIO_OT_show_log(bpy.types.Operator):
    bl_idname = "inspatio.show_log"
    bl_label = "Journal"
    bl_description = "Affiche le journal complet dans un éditeur de texte"

    def execute(self, context):
        text = bpy.data.texts.get(LOG_TEXT)
        if text is None:
            self.report({"INFO"}, "Journal vide")
            return {"CANCELLED"}
        bpy.ops.wm.window_new()
        area = context.window_manager.windows[-1].screen.areas[0]
        area.type = "TEXT_EDITOR"
        area.spaces.active.text = text
        return {"FINISHED"}


# --------------------------------------------------------------------------
# Properties and UI

class InSpatioPreferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_KEY

    mode: bpy.props.EnumProperty(
        name="Exécution",
        items=(("LOCAL", "Linux (local)", "InSpatio est installé sur la même machine Linux que Blender"),
               ("WSL", "Windows + WSL", "Blender sous Windows, InSpatio installé dans Ubuntu (WSL)")),
        default="WSL" if sys.platform == "win32" else "LOCAL")
    repo_path: bpy.props.StringProperty(
        name="Dossier In-spatio", subtype="DIR_PATH",
        description="Dossier du dépôt (sous WSL : chemin Linux, ex. /home/moi/In-spatio)")
    python_path: bpy.props.StringProperty(
        name="Python InSpatio",
        description="Résultat de « conda activate inspatio_world_test && which python »")
    wsl_distro: bpy.props.StringProperty(name="Distribution WSL", default="Ubuntu-24.04",
                                         description="Nom affiché par « wsl -l »")
    gpu: bpy.props.StringProperty(name="Carte graphique", default="",
                                  description="Vide = automatique, sinon numéro (0, 1…)")

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "mode", expand=True)
        layout.prop(self, "repo_path")
        layout.prop(self, "python_path")
        if self.mode == "WSL":
            layout.prop(self, "wsl_distro")
        layout.prop(self, "gpu")
        box = layout.box()
        box.label(text="Python InSpatio : dans un terminal "
                       f"{'Ubuntu ' if self.mode == 'WSL' else ''}tape")
        box.label(text="conda activate inspatio_world_test && which python")
        problem = check_setup()
        if problem:
            layout.label(text=problem, icon="ERROR")
        else:
            layout.label(text="Configuration trouvée", icon="CHECKMARK")


class InSpatioSceneProperties(bpy.types.PropertyGroup):
    source_mode: bpy.props.EnumProperty(
        name="Source",
        items=(("SCENE", "Dossier de scène", "Un exemple (examples/…) ou une scène préparée"),
               ("VIDEO", "Vidéo", "Une vidéo : profondeur et caméras estimées automatiquement")),
        default="SCENE")
    scene_dir: bpy.props.StringProperty(name="Dossier", subtype="DIR_PATH")
    video_path: bpy.props.StringProperty(name="Vidéo", subtype="FILE_PATH")
    new_name: bpy.props.StringProperty(name="Nom", description="Nom de la scène (vide = nom du fichier)")
    prompt: bpy.props.StringProperty(name="Description", description="Description de la scène, en anglais")
    mesh_step: bpy.props.IntProperty(name="Détail 3D (pixels)", default=4, min=1, max=16,
                                     description="Un point tous les N pixels pour l'aperçu 3D")
    video_mesh_frames: bpy.props.IntProperty(name="Images en 3D", default=1, min=1, max=16,
                                             description="Nombre d'images de la vidéo montrées en 3D")
    scene_name: bpy.props.StringProperty()
    kind: bpy.props.StringProperty()
    frames: bpy.props.IntProperty()
    lens: bpy.props.FloatProperty()
    status: bpy.props.StringProperty()
    error: bpy.props.StringProperty()
    result_path: bpy.props.StringProperty()


class INSPATIO_PT_panel(bpy.types.Panel):
    bl_label = "InSpatio-World"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "InSpatio"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.inspatio
        problem = check_setup()
        if problem:
            layout.label(text=problem, icon="ERROR")

        box = layout.box()
        box.label(text="1. Scène", icon="SCENE_DATA")
        box.row().prop(props, "source_mode", expand=True)
        box.prop(props, "video_path" if props.source_mode == "VIDEO" else "scene_dir")
        box.prop(props, "new_name")
        if props.source_mode == "VIDEO":
            box.prop(props, "prompt")
            box.prop(props, "video_mesh_frames")
        box.prop(props, "mesh_step")
        box.operator("inspatio.prepare", icon="IMPORT")

        box = layout.box()
        box.label(text="2. Caméra", icon="CAMERA_DATA")
        if props.scene_name:
            box.label(text=f"{props.scene_name} : {props.kind}, {props.frames} images")
            if props.kind == "video":
                box.label(text=f"Plage obligatoire : {scene.frame_start} – {scene.frame_start + props.frames - 1}")
                box.label(text=f"La caméra suit « {SOURCE_RIG_NAME} » :")
                box.label(text="décale-la, ou Alt+P pour la libérer")
            else:
                box.label(text="Anime InSpatio_Camera (touche I)")
                box.label(text="La plage d'images = la durée")
            box.operator("inspatio.view_camera", icon="VIEW_CAMERA")
        else:
            box.label(text="Prépare d'abord une scène")

        box = layout.box()
        box.label(text="3. Génération", icon="RENDER_ANIMATION")
        box.prop(props, "prompt")
        row = box.row()
        row.enabled = bool(props.scene_name) and not job_running()
        row.operator("inspatio.generate", icon="PLAY")
        if props.result_path:
            box.operator("inspatio.open_result", icon="FILE_MOVIE")

        if job_running() or props.status:
            box = layout.box()
            box.label(text=props.status or "…", icon="TIME" if job_running() else "INFO")
            if props.error:
                for chunk in range(0, min(len(props.error), 240), 60):
                    box.label(text=props.error[chunk:chunk + 60], icon="ERROR" if chunk == 0 else "BLANK1")
            row = box.row()
            if job_running():
                row.operator("inspatio.cancel", icon="CANCEL")
            row.operator("inspatio.show_log", icon="TEXT")


classes = (InSpatioPreferences, InSpatioSceneProperties, INSPATIO_OT_prepare, INSPATIO_OT_generate,
           INSPATIO_OT_cancel, INSPATIO_OT_view_camera, INSPATIO_OT_open_result, INSPATIO_OT_show_log,
           INSPATIO_PT_panel)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.inspatio = bpy.props.PointerProperty(type=InSpatioSceneProperties)


def unregister():
    if bpy.app.timers.is_registered(poll_job):
        bpy.app.timers.unregister(poll_job)
    del bpy.types.Scene.inspatio
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
