# SculptForge - node tests against the real sculptforge-cli (skipped when it isn't installed).
# SPDX-License-Identifier: GPL-3.0-or-later
import os

import numpy as np
import pytest
import torch
from PIL import Image

from sculptforge_comfy import cli, meshio, nodes

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")


@pytest.fixture(autouse=True)
def real_cli():
    try:
        cli.find_cli()
    except cli.SculptForgeError as e:
        pytest.skip(str(e))


def load_teddy():
    """examples/teddy.png the way ComfyUI's LoadImage returns it."""
    rgba = np.asarray(Image.open(os.path.join(EXAMPLES, "teddy.png")).convert("RGBA"), dtype=np.float32) / 255
    return torch.from_numpy(rgba[..., :3])[None], torch.from_numpy(1.0 - rgba[..., 3])[None]


def triangles(mesh):
    text = nodes.SculptForgeMeshInfo().info(mesh)["result"][0]
    return int(text.split(" triangles")[0].split()[-1])


def test_teddy_to_game_ready_assets(tmp_path):
    image, mask = load_teddy()
    (mesh,) = nodes.SculptForgeInflateImage().inflate(image, 1.0, 128, 1, mask)
    inflated = triangles(mesh)
    assert inflated > 10_000
    (mesh,) = nodes.SculptForgeVoxelRemesh().remesh(mesh, 96)
    (mesh,) = nodes.SculptForgeDecimate().decimate(mesh, 0.5, 0)
    decimated = triangles(mesh)
    (sub,) = nodes.SculptForgeSubdivide().subdivide(mesh, 1)
    assert triangles(sub) == decimated * 4

    glb = nodes.SculptForgeExport().export(mesh, "glb", "test/teddy")["result"][0]
    assert meshio.glb_bytes_magic(glb)
    for fmt in ("fbx", "usdz", "obj", "stl"):
        path = nodes.SculptForgeExport().export(mesh, fmt, "test/teddy")["result"][0]
        assert os.path.getsize(path) > 1000, fmt
    lods = nodes.SculptForgeLODChain().export(mesh, 3, 0.5, "test/teddy_lods")["result"][0].splitlines()
    assert len(lods) == 3 and all(meshio.glb_bytes_magic(p) for p in lods)


def test_core_mesh_roundtrip_keeps_geometry_and_colour():
    v = torch.tensor([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=torch.float32)
    f = torch.tensor([[[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]], dtype=torch.int64)
    core = meshio.mesh_class()(vertices=v, faces=f, vertex_colors=torch.full((1, 4, 3), 0.5))
    (sf,) = nodes.SculptForgeSubdivide().subdivide(core, 1)
    (back,) = nodes.SculptForgeToMesh().convert(sf)
    assert tuple(back.faces.shape) == (1, 16, 3) and back.vertices.shape[1] == 10
    np.testing.assert_allclose(back.vertex_colors[0].numpy(), 0.5, atol=1 / 255)
    assert back.vertices.min() >= -1e-6 and back.vertices.max() <= 1 + 1e-6
