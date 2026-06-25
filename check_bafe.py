import torch

from ultralytics.nn.tasks import DetectionModel


model = DetectionModel("ultralytics/cfg/models/11/yolo11-bafe.yaml", nc=80, verbose=False)
output = model(torch.zeros(1, 3, 640, 640))

assert model.stride.tolist() == [8.0, 16.0, 32.0]
assert output is not None
print("BAFE model checks passed.")
