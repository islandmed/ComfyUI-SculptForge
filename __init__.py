# ComfyUI-SculptForge: SculptForge nodes for ComfyUI (remesh, decimate, LODs, export to GLB/FBX/USD).
# SPDX-License-Identifier: GPL-3.0-or-later
if __package__:  # loaded by ComfyUI as a package (pytest imports this file on its own)
    from .sculptforge_comfy import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
