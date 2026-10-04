from types import SimpleNamespace

import pytest
import torch
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.tasks import DetectionModel

from sdcl.config import SDCLConfig
from sdcl.criterion import SDCLLoss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_amp_forward_backward_nonzero_weights():
    torch.manual_seed(0)
    model = DetectionModel("yolo11n.yaml", nc=2, verbose=False).cuda().train()
    model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
    model.sdcl_settings = SDCLConfig(warmup_epochs=0, ramp_epochs=0, log_interval=1).to_dict()
    model.sdcl_ignore_regions = True
    criterion = SDCLLoss(model)
    batch = {
        "img": torch.rand(2, 3, 128, 128, device="cuda"),
        "batch_idx": torch.tensor([0.0, 0.0, 1.0], device="cuda"),
        "cls": torch.tensor([[0.0], [1.0], [0.0]], device="cuda"),
        "bboxes": torch.tensor([[0.3, 0.3, 0.1, 0.1], [0.7, 0.7, 0.3, 0.3],
                                [0.4, 0.4, 0.15, 0.15]], device="cuda"),
        "ignore_bboxes": torch.tensor([[0.9, 0.9, 0.1, 0.1]], device="cuda"),
        "ignore_batch_idx": torch.tensor([0], device="cuda"),
    }
    with torch.autocast("cuda", dtype=torch.float16):
        prediction = model(batch["img"])
        loss = criterion(prediction, batch)[0].sum()
    # Random initialization can overflow at the default 65536 scale even upstream.
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0)
    scaler.scale(loss).backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert criterion.diagnostics["strength"] == 0.5
    assert criterion.diagnostics["matched_objects"] > 0
    assert criterion.diagnostics["mass_relative_error"] < 1e-5
