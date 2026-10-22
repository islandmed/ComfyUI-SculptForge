# SculptForge - ComfyUI nodes: ComfyUI folders (input, output, temp) and UI helpers.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Inside ComfyUI these use folder_paths and PromptServer; without them (unit tests, scripts)
# they fall back to folders under the system temp directory.
import os
import tempfile

try:
    import folder_paths
except ImportError:  # not running inside ComfyUI
    folder_paths = None


def _fallback(name):
    path = os.path.join(tempfile.gettempdir(), "comfyui-sculptforge", name)
    os.makedirs(path, exist_ok=True)
    return path


def input_dir():
    return folder_paths.get_input_directory() if folder_paths else _fallback("input")


def output_dir():
    return folder_paths.get_output_directory() if folder_paths else _fallback("output")


def work_dir():
    """Where intermediate meshes live: ComfyUI's temp folder (cleared when ComfyUI starts)."""
    base = folder_paths.get_temp_directory() if folder_paths else _fallback("temp")
    path = os.path.join(base, "sculptforge")
    os.makedirs(path, exist_ok=True)
    return path


def resolve_input(path):
    path = os.path.expanduser(path.strip().strip('"'))
    if not path:
        raise ValueError("No file path given")
    return path if os.path.isabs(path) else os.path.join(input_dir(), path)


def save_path(filename_prefix):
    """(folder, filename, counter, subfolder) for a new output file, numbered like ComfyUI's save nodes."""
    if folder_paths:
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, output_dir())
        os.makedirs(folder, exist_ok=True)
        return folder, filename, counter, subfolder
    root = output_dir()
    subfolder = os.path.dirname(os.path.normpath(filename_prefix))
    filename = os.path.basename(os.path.normpath(filename_prefix))
    folder = os.path.abspath(os.path.join(root, subfolder))
    if os.path.commonpath([folder, os.path.abspath(root)]) != os.path.abspath(root):
        raise ValueError("Saving outside the output folder is not allowed")
    os.makedirs(folder, exist_ok=True)
    taken = [f[len(filename) + 1:].split("_")[0].split(".")[0] for f in os.listdir(folder)
             if f.startswith(filename + "_")]
    counter = max([int(t) for t in taken if t.isdigit()], default=0) + 1
    return folder, filename, counter, subfolder


def show_text(text, unique_id):
    """Show text on the node in the ComfyUI frontend (like the core Get Mesh Info node)."""
    if unique_id is None:
        return
    try:
        from server import PromptServer
        PromptServer.instance.send_progress_text(text, unique_id)
    except Exception:
        pass
