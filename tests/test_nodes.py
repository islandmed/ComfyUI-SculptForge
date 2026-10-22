# SculptForge - ComfyUI node tests against the fake CLI (no SculptForge install needed).
# SPDX-License-Identifier: GPL-3.0-or-later
import os
import struct

import numpy as np
import pytest
import torch
from PIL import Image

from sculptforge_comfy import cli, meshio, nodes, paths


class CoreMesh:
    """Shaped like comfy_api.latest.Types.MESH."""

    def __init__(self, vertices, faces, vertex_colors=None, vertex_counts=None, face_counts=None):
        self.vertices, self.faces, self.vertex_colors = vertices, faces, vertex_colors
        self.vertex_counts, self.face_counts = vertex_counts, face_counts


def tetra_mesh():
    v = torch.tensor([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=torch.float32)
    f = torch.tensor([[[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]], dtype=torch.int64)
    return CoreMesh(v, f)


def disc_image(size=64):
    """An orange disc on a transparent background, as LoadImage returns it (IMAGE + MASK)."""
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    inside = ((yy - size / 2) ** 2 + (xx - size / 2) ** 2) < (size / 3) ** 2
    image = torch.zeros(1, size, size, 3)
    image[0][inside] = torch.tensor([1.0, 0.5, 0.0])
    mask = (~inside).float()[None]
    return image, mask


# ---------------------------------------------------------------- finding and running the CLI


def test_cli_env_var_pointing_nowhere_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv(cli.CLI_ENV, str(tmp_path / "missing"))
    with pytest.raises(cli.SculptForgeError, match="SCULPTFORGE_CLI is set"):
        cli.find_cli()


def test_cli_missing_explains_how_to_fix(monkeypatch):
    monkeypatch.delenv(cli.CLI_ENV, raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(cli, "_install_dirs", lambda: [])
    with pytest.raises(cli.SculptForgeError, match="SCULPTFORGE_CLI"):
        cli.find_cli()


def test_cli_found_in_install_dir_or_env_folder(monkeypatch, fake_cli):
    folder = os.path.dirname(fake_cli.path)
    monkeypatch.setenv(cli.CLI_ENV, folder)  # a folder works too
    assert cli.find_cli() == fake_cli.path
    monkeypatch.delenv(cli.CLI_ENV)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(cli, "_install_dirs", lambda: ["/nonexistent", folder])
    assert cli.find_cli() == fake_cli.path


def test_paths_go_through_variables_not_script_text(fake_cli, tmp_path):
    odd = tmp_path / 'a folder; with "quotes" and $dollar'
    odd.mkdir()
    src = odd / "in put.obj"
    src.write_text('{"name": "rock", "triangles": 100}')
    out = odd / "out.sfproj"
    cli.run("open path=$IN; save_project path=$OUT", {"IN": str(src), "OUT": str(out)})
    call = fake_cli()[-1]
    assert call["script"] == "open path=$IN; save_project path=$OUT"
    assert call["vars"] == {"IN": str(src), "OUT": str(out)}
    assert out.exists()


def test_cli_errors_carry_the_cli_message(fake_cli, monkeypatch):
    monkeypatch.setenv("FAKE_SF_FAIL", "Resolution too high")
    with pytest.raises(cli.SculptForgeError, match="Resolution too high"):
        cli.run("new_sphere")


# ---------------------------------------------------------------- nodes


def test_every_node_is_well_formed():
    assert set(nodes.NODE_CLASS_MAPPINGS) == set(nodes.NODE_DISPLAY_NAME_MAPPINGS)
    for name, cls in nodes.NODE_CLASS_MAPPINGS.items():
        assert cls.CATEGORY == "SculptForge", name
        assert callable(getattr(cls, cls.FUNCTION)), name
        spec = cls.INPUT_TYPES()
        assert "required" in spec, name
        assert len(getattr(cls, "RETURN_NAMES", cls.RETURN_TYPES)) == len(cls.RETURN_TYPES), name
        assert cls.DESCRIPTION, name
    # Mesh inputs accept SculptForge meshes and ComfyUI's core MESH.
    assert nodes.SculptForgeVoxelRemesh.INPUT_TYPES()["required"]["mesh"][0] == "SF_MESH,MESH"


def test_inflate_writes_png_with_mask_as_alpha(fake_cli):
    image, mask = disc_image()
    (mesh,) = nodes.SculptForgeInflateImage().inflate(image, 0.8, 96, 2, mask)
    assert isinstance(mesh, nodes.SFMesh) and os.path.isfile(mesh.path)
    call = fake_cli()[-1]
    assert call["script"].startswith("image_inflate path=$IMG thickness=0.8 resolution=96 subdivisions=2;")
    png = Image.open(call["vars"]["IMG"])
    assert png.mode == "RGBA"
    alpha = np.asarray(png)[..., 3]
    assert alpha[32, 32] == 255 and alpha[0, 0] == 0


def test_inflate_ignores_empty_mask(fake_cli):
    image, _ = disc_image()
    nodes.SculptForgeInflateImage().inflate(image, 1.0, 160, 1, torch.zeros(1, 64, 64))
    assert Image.open(fake_cli()[-1]["vars"]["IMG"]).mode == "RGB"


def test_processing_chain(fake_cli):
    image, mask = disc_image()
    (mesh,) = nodes.SculptForgeInflateImage().inflate(image, 1.0, 160, 1, mask)
    (mesh,) = nodes.SculptForgeVoxelRemesh().remesh(mesh, 128)
    assert mesh.info == "Remeshed to 1280 triangles"
    (mesh,) = nodes.SculptForgeDecimate().decimate(mesh, 0.25, 0)
    assert mesh.info == "Decimated to 320 triangles"
    (mesh,) = nodes.SculptForgeDecimate().decimate(mesh, 0.25, 100)
    assert mesh.info == "Decimated to 100 triangles"
    (mesh,) = nodes.SculptForgeSubdivide().subdivide(mesh, 2)
    assert mesh.info == "Subdivided to 1600 triangles"
    scripts = [c["script"] for c in fake_cli()]
    assert scripts[1] == "open path=$IN; remesh resolution=128; save_project path=$OUT"
    assert "decimate ratio=0.25;" in scripts[2] and "decimate triangles=100;" in scripts[3]
    # Each step reads the previous step's output.
    calls = fake_cli()
    assert all(calls[i + 1]["vars"]["IN"] == calls[i]["vars"]["OUT"] for i in range(len(calls) - 1))


def test_core_mesh_input_is_converted_to_ply(fake_cli):
    # A zero-padded batch of two: only the first item's real vertices/faces may be used.
    v = torch.zeros(2, 6, 3)
    v[0, :4] = tetra_mesh().vertices[0]
    f = torch.zeros(2, 5, 3, dtype=torch.int64)
    f[0, :4] = tetra_mesh().faces[0]
    colors = torch.ones(2, 6, 3) * torch.tensor([1.0, 0.0, 0.0])
    mesh = CoreMesh(v, f, colors, torch.tensor([4, 6]), torch.tensor([4, 5]))
    (out,) = nodes.SculptForgeVoxelRemesh().remesh(mesh, 64)
    ply = fake_cli()[-1]["vars"]["IN"]
    assert os.path.basename(ply) == "ComfyUI_mesh.ply"
    m = meshio.read_ply(ply)
    assert m["vertices"].shape == (4, 3) and m["faces"].shape == (4, 3)
    np.testing.assert_allclose(m["colors"][0], [1, 0, 0])
    assert out.info == "Remeshed to 640 triangles"


def test_to_mesh_produces_core_mesh(fake_cli):
    (mesh,) = nodes.SculptForgeVoxelRemesh().remesh(tetra_mesh(), 64)
    (core,) = nodes.SculptForgeToMesh().convert(mesh)
    assert tuple(core.vertices.shape) == (1, 4, 3) and core.vertices.dtype == torch.float32
    assert tuple(core.faces.shape) == (1, 4, 3) and core.faces.dtype == torch.int64
    assert tuple(core.normals.shape) == (1, 4, 3)
    np.testing.assert_allclose(core.vertex_colors[0, 0].numpy(), [1.0, 128 / 255, 0.0], atol=1e-6)


@pytest.mark.parametrize("fmt,command", sorted(nodes.EXPORT_COMMANDS.items()))
def test_export_formats(fake_cli, comfy_dirs, fmt, command):
    (mesh,) = nodes.SculptForgeVoxelRemesh().remesh(tetra_mesh(), 64)
    result = nodes.SculptForgeExport().export(mesh, fmt, "3d/Rock")
    entry = result["ui"]["3d"][0]
    assert entry == {"filename": f"Rock_00001_.{fmt}", "subfolder": "3d", "type": "output"}
    path = result["result"][0]
    assert path == str(comfy_dirs / "output" / "3d" / entry["filename"]) and os.path.isfile(path)
    assert fake_cli()[-1]["script"] == f"open path=$IN; {command} path=$OUT"


def test_export_numbers_files_like_comfyui(fake_cli):
    (mesh,) = nodes.SculptForgeVoxelRemesh().remesh(tetra_mesh(), 64)
    names = [nodes.SculptForgeExport().export(mesh, "glb", "Rock")["ui"]["3d"][0]["filename"] for _ in range(3)]
    assert names == ["Rock_00001_.glb", "Rock_00002_.glb", "Rock_00003_.glb"]


def test_export_refuses_paths_outside_output(fake_cli):
    (mesh,) = nodes.SculptForgeVoxelRemesh().remesh(tetra_mesh(), 64)
    with pytest.raises(ValueError):
        nodes.SculptForgeExport().export(mesh, "glb", "../../escape")


def test_lod_chain(fake_cli, comfy_dirs):
    (mesh,) = nodes.SculptForgeVoxelRemesh().remesh(tetra_mesh(), 64)
    result = nodes.SculptForgeLODChain().export(mesh, 3, 0.4, "lods/Rock")
    names = [e["filename"] for e in result["ui"]["3d"]]
    assert names == ["Rock_00001_LOD0.glb", "Rock_00001_LOD1.glb", "Rock_00001_LOD2.glb"]
    assert all(meshio.glb_bytes_magic(p) for p in result["result"][0].splitlines())
    assert "export_lods path=$OUT levels=3 ratio=0.4" in fake_cli()[-1]["script"]
    # The next chain gets the next number.
    again = nodes.SculptForgeLODChain().export(mesh, 1, 0.5, "lods/Rock")
    assert again["ui"]["3d"][0]["filename"] == "Rock_00002_LOD0.glb"


def test_mesh_info(fake_cli):
    (mesh,) = nodes.SculptForgeVoxelRemesh().remesh(tetra_mesh(), 64)
    result = nodes.SculptForgeMeshInfo().info(mesh)
    text = result["result"][0]
    assert text.startswith("320 vertices, 640 triangles")
    assert "ComfyUI_mesh (640 triangles, active)" in text
    assert result["ui"]["text"] == (text,)


def test_load_mesh_relative_to_input(fake_cli, comfy_dirs):
    (comfy_dirs / "input" / "3d").mkdir(parents=True)
    (comfy_dirs / "input" / "3d" / "rock.obj").write_text('{"name": "rock", "triangles": 100}')
    (mesh,) = nodes.SculptForgeLoadMesh().load("3d/rock.obj")
    assert fake_cli()[-1]["vars"]["IN"] == str(comfy_dirs / "input" / "3d" / "rock.obj")
    assert os.path.isfile(mesh.path)
    assert nodes.SculptForgeLoadMesh.IS_CHANGED("3d/rock.obj") > 0
    with pytest.raises(cli.SculptForgeError, match="not found"):
        nodes.SculptForgeLoadMesh().load("3d/missing.obj")


def test_missing_working_file_is_explained(fake_cli):
    with pytest.raises(cli.SculptForgeError, match="temp folder"):
        nodes.SculptForgeDecimate().decimate(nodes.SFMesh("/nonexistent/x.sfproj"), 0.5, 0)


def test_open_in_app(fake_cli, monkeypatch):
    launched = []
    monkeypatch.setattr(cli, "launch_app", lambda path: launched.append(path) or "sculptforge")
    node = nodes.SculptForgeOpenInApp()
    assert node.open(False, file_path="/x.glb") == {}
    node.open(True, file_path="/out/a.glb\n/out/b.glb")
    (mesh,) = nodes.SculptForgeVoxelRemesh().remesh(tetra_mesh(), 64)
    node.open(True, mesh=mesh)
    assert launched[0] == "/out/a.glb"
    assert launched[1].endswith("SculptForge from ComfyUI.sfproj") and os.path.isfile(launched[1])
    with pytest.raises(cli.SculptForgeError):
        node.open(True)


def test_launch_app_finds_app_next_to_cli(fake_cli, monkeypatch, tmp_path):
    app = os.path.join(os.path.dirname(fake_cli.path), "sculptforge")
    with open(app, "w") as f:
        f.write("")
    monkeypatch.delenv(cli.APP_ENV, raising=False)
    assert cli.find_app() == app
    started = []
    monkeypatch.setattr(cli.subprocess, "Popen", lambda args, **kw: started.append((args, kw)))
    cli.launch_app("/tmp/a.glb")
    assert started[0][0] == [app, "/tmp/a.glb"] and "shell" not in started[0][1]


# ---------------------------------------------------------------- PLY reading/writing


def test_ply_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    v = rng.random((50, 3), dtype=np.float32)
    f = rng.integers(0, 50, (80, 3))
    c = rng.random((50, 3))
    path = tmp_path / "m.ply"
    meshio.write_ply(path, v, f, c)
    m = meshio.read_ply(path)
    np.testing.assert_array_equal(m["vertices"], v)
    np.testing.assert_array_equal(m["faces"], f)
    np.testing.assert_allclose(m["colors"], c, atol=0.5 / 255 + 1e-6)
    assert m["normals"] is None


def test_ply_polygons_are_fan_triangulated(tmp_path):
    header = (b"ply\nformat binary_little_endian 1.0\nelement vertex 5\nproperty float x\nproperty float y\n"
              b"property float z\nelement face 2\nproperty list uchar uint vertex_indices\nend_header\n")
    body = b"".join(struct.pack("<fff", *p) for p in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (2, 0, 0)])
    body += struct.pack("<BIIII", 4, 0, 1, 2, 3) + struct.pack("<BIII", 3, 1, 4, 2)
    path = tmp_path / "quad.ply"
    path.write_bytes(header + body)
    np.testing.assert_array_equal(meshio.read_ply(path)["faces"], [[0, 1, 2], [0, 2, 3], [1, 4, 2]])


def test_ply_rejects_out_of_range_faces(tmp_path):
    with pytest.raises(ValueError):
        meshio.write_ply(tmp_path / "bad.ply", np.zeros((3, 3)), [[0, 1, 3]])


def test_save_path_uses_comfyui_when_available(monkeypatch, tmp_path):
    class FolderPaths:
        @staticmethod
        def get_output_directory():
            return str(tmp_path)

        @staticmethod
        def get_save_image_path(prefix, output_dir):
            return str(tmp_path / "sub"), "name", 7, "sub", prefix

    monkeypatch.setattr(paths, "folder_paths", FolderPaths)
    assert paths.save_path("sub/name") == (str(tmp_path / "sub"), "name", 7, "sub")
    assert os.path.isdir(tmp_path / "sub")
