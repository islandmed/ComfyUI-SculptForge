# SculptForge - end-to-end test: a real ComfyUI server with this pack and the real sculptforge-cli.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Starts its own ComfyUI (CPU, port $SF_COMFYUI_PORT or 8189) with a throwaway base directory
# whose custom_nodes/ holds a symlink to this pack, checks the nodes register, then queues the
# example workflows through the HTTP API and opens the exported files with sculptforge-cli.
#
#   COMFYUI_DIR=/path/to/ComfyUI [COMFYUI_PYTHON=/path/to/venv/bin/python] \
#   SCULPTFORGE_CLI=/path/to/sculptforge-cli pytest tests/test_comfyui_e2e.py
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

import pytest

from sculptforge_comfy import cli, meshio, nodes

PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(PACK_DIR, "examples")
COMFYUI_DIR = os.environ.get("COMFYUI_DIR")
PORT = int(os.environ.get("SF_COMFYUI_PORT", "8189"))
BASE = f"http://127.0.0.1:{PORT}"

pytestmark = pytest.mark.skipif(not COMFYUI_DIR, reason="set COMFYUI_DIR to a ComfyUI checkout to run")


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return r.read()


def get_json(path):
    return json.loads(get(path))


def post_json(path, payload):
    req = urllib.request.Request(BASE + path, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise AssertionError(f"POST {path} failed: {e.code} {e.read().decode(errors='replace')}") from None


def upload_image(path):
    boundary = uuid.uuid4().hex
    with open(path, "rb") as f:
        data = f.read()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{os.path.basename(path)}\"\r\n"
            "Content-Type: image/png\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(BASE + "/upload/image", body,
                                 {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["name"]


def run_workflow(name, image):
    with open(os.path.join(EXAMPLES, name), encoding="utf-8") as f:
        prompt = json.load(f)
    for node in prompt.values():
        if node["class_type"] == "LoadImage":
            node["inputs"]["image"] = image
    prompt_id = post_json("/prompt", {"prompt": prompt, "client_id": "sculptforge-e2e"})["prompt_id"]
    deadline = time.time() + 300
    while time.time() < deadline:
        history = get_json(f"/history/{prompt_id}")
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error" or not status.get("completed", True):
                messages = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                raise AssertionError(f"{name} failed: {json.dumps(messages, indent=1)[:4000]}")
            return entry["outputs"]
        time.sleep(0.5)
    raise AssertionError(f"{name} did not finish in time")


def download(entry, folder):
    query = urllib.parse.urlencode({k: entry[k] for k in ("filename", "subfolder", "type")})
    path = os.path.join(folder, entry["filename"])
    with open(path, "wb") as f:
        f.write(get(f"/view?{query}"))
    return path


def triangles(path):
    out = cli.run("open path=$IN; mesh_info", {"IN": path})
    info = out.strip().splitlines()[-1]
    return int(info.split(" triangles")[0].split()[-1])


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    try:
        cli.find_cli()
    except cli.SculptForgeError as e:
        pytest.skip(str(e))
    base = tmp_path_factory.mktemp("comfy-base")
    (base / "custom_nodes").mkdir()
    os.symlink(PACK_DIR, base / "custom_nodes" / "ComfyUI-SculptForge")
    python = os.environ.get("COMFYUI_PYTHON") or sys.executable
    log = open(base / "server.log", "w")
    proc = subprocess.Popen(
        [python, "main.py", "--cpu", "--listen", "127.0.0.1", "--port", str(PORT),
         "--base-directory", str(base), "--disable-auto-launch"],
        cwd=COMFYUI_DIR, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, SCULPTFORGE_CLI=cli.find_cli()))
    try:
        deadline = time.time() + 180
        while True:
            try:
                get("/system_stats")
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                if proc.poll() is not None or time.time() > deadline:
                    raise AssertionError("ComfyUI did not start:\n" + (base / "server.log").read_text()[-4000:])
                time.sleep(1)
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


def test_nodes_register(server):
    info = get_json("/object_info")
    for name, display in nodes.NODE_DISPLAY_NAME_MAPPINGS.items():
        assert name in info, f"{name} missing; see {server / 'server.log'}"
        assert info[name]["display_name"] == display
        assert info[name]["category"] == "SculptForge"
    assert info["SculptForgeVoxelRemesh"]["input"]["required"]["mesh"][0] == "SF_MESH,MESH"
    assert info["SculptForgeToMesh"]["output"] == ["MESH"]
    assert info["SculptForgeExport"]["output_node"] is True
    log = (server / "server.log").read_text()
    assert "IMPORT FAILED" not in log or "ComfyUI-SculptForge" not in log


def test_inflate_remesh_export_glb(server, tmp_path):
    image = upload_image(os.path.join(EXAMPLES, "teddy.png"))
    outputs = run_workflow("inflate-remesh-export.json", image)
    entries = outputs["4"]["3d"]
    assert len(entries) == 1 and entries[0]["type"] == "output" and entries[0]["filename"].endswith(".glb")
    glb = download(entries[0], tmp_path)
    assert meshio.glb_bytes_magic(glb)
    assert triangles(glb) > 10_000
    shutil.copy(glb, server / "teddy-e2e.glb")  # kept with the server log for inspection


def test_core_mesh_roundtrip(server, tmp_path):
    image = upload_image(os.path.join(EXAMPLES, "teddy.png"))
    outputs = run_workflow("core-mesh-roundtrip.json", image)
    # SculptForge -> core MESH -> core SaveGLB
    core_glb = download(outputs["5"]["3d"][0], tmp_path)
    assert meshio.glb_bytes_magic(core_glb) and triangles(core_glb) > 10_000
    # core MESH -> SculptForge remesh/decimate -> LODs and FBX
    lods = [download(e, tmp_path) for e in outputs["9"]["3d"]]
    counts = [triangles(p) for p in lods]
    assert len(counts) == 3 and counts[0] > counts[1] > counts[2] > 0
    fbx = download(outputs["10"]["3d"][0], tmp_path)
    assert fbx.endswith(".fbx") and triangles(fbx) == counts[0]
    info = outputs["8"]["text"][0]
    assert f"{counts[0]} triangles" in info
