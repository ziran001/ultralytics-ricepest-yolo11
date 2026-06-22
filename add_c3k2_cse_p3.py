from pathlib import Path
import shutil

root = Path("/root/ultralytics-8.3.27")
block = root / "ultralytics/nn/modules/block.py"
init = root / "ultralytics/nn/modules/__init__.py"
tasks = root / "ultralytics/nn/tasks.py"
yolo11 = root / "ultralytics/cfg/models/11/yolo11.yaml"
new_yaml = root / "ultralytics/cfg/models/11/yolo11-cse-p3.yaml"

for p in [block, init, tasks]:
    shutil.copy2(p, p.with_suffix(p.suffix + ".bak_c3k2_cse"))

text = block.read_text(encoding="utf-8")

text = text.replace(
    "from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad",
    "from .conv import ChannelAttention, Conv, DWConv, GhostConv, LightConv, RepConv, SpatialAttention, autopad",
)

if '"C3k2CSE",' not in text:
    text = text.replace('    "C3k2",\n', '    "C3k2",\n    "C3k2CSE",\n', 1)

if "class C3k2CSE(C3k2):" not in text:
    marker = "\n\nclass C3k("
    cse = r'''

class C3k2CSE(C3k2):
    """C3k2 with lightweight channel-spatial enhancement for small-object features."""

    def __init__(
        self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True
    ):
        """
        Initialize C3k2CSE.

        This module keeps the original C3k2 feature extractor and adds channel attention followed by 7x7 spatial
        attention. The enhanced feature is added back to the C3k2 output as a local residual.
        """
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.channel_attn = ChannelAttention(c2)
        self.spatial_attn = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C3k2 followed by channel-spatial enhancement."""
        y = super().forward(x)
        return y + self.spatial_attn(self.channel_attn(y))
'''
    text = text.replace(marker, cse + marker, 1)

block.write_text(text, encoding="utf-8")

text = init.read_text(encoding="utf-8")
if "C3k2CSE," not in text:
    text = text.replace("    C3k2,\n", "    C3k2,\n    C3k2CSE,\n", 1)
if '"C3k2CSE",' not in text:
    text = text.replace('    "C3k2",\n', '    "C3k2",\n    "C3k2CSE",\n', 1)
init.write_text(text, encoding="utf-8")

text = tasks.read_text(encoding="utf-8")
if "C3k2CSE," not in text:
    text = text.replace("    C3k2,\n", "    C3k2,\n    C3k2CSE,\n", 1)
if "C3k2CSE" not in text.split("base_modules", 1)[1].split("repeat_modules", 1)[0]:
    text = text.replace("            C3k2,\n", "            C3k2,\n            C3k2CSE,\n", 1)
if "C3k2CSE" not in text.split("repeat_modules", 1)[1].split("for i,", 1)[0]:
    text = text.replace("            C3k2,\n", "            C3k2,\n            C3k2CSE,\n", 1)
text = text.replace("if m is C3k2:", "if m in frozenset({C3k2, C3k2CSE}):")
tasks.write_text(text, encoding="utf-8")

if not yolo11.exists():
    raise FileNotFoundError(f"Cannot find {yolo11}")

yaml = yolo11.read_text(encoding="utf-8")
yaml = yaml.replace(
    "# YOLO11 object detection model with P3/8 - P5/32 outputs",
    "# YOLO11 object detection model with C3k2CSE at the backbone P3/8 stage",
    1,
)
yaml = yaml.replace(
    "  - [-1, 2, C3k2, [512, False, 0.25]]\n  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16",
    "  - [-1, 2, C3k2CSE, [512, False, 0.25]] # 4: CSE on backbone P3/8\n  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16",
    1,
)
new_yaml.write_text(yaml, encoding="utf-8")

print("Done.")
print(f"Created: {new_yaml}")
print("Use model: ultralytics/cfg/models/11/yolo11-cse-p3.yaml")
