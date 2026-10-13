"""Export predictions for evaluation with the official VisDrone toolkit."""

from pathlib import Path


def export_visdrone(model_path, images, output, device="0", imgsz=640):
    from ultralytics import YOLO

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite predictions: {output}")
    output.mkdir(parents=True)
    model = YOLO(str(model_path))
    results = model.predict(
        source=str(images), stream=True, imgsz=imgsz, device=device,
        conf=0.001, iou=0.7, max_det=500, verbose=False,
    )
    count = 0
    for result in results:
        lines = []
        for box, score, cls in zip(result.boxes.xyxy.cpu(), result.boxes.conf.cpu(), result.boxes.cls.cpu()):
            x1, y1, x2, y2 = box.tolist()
            lines.append(f"{x1:.4f},{y1:.4f},{x2-x1:.4f},{y2-y1:.4f},{float(score):.6f},{int(cls)+1},0,0")
        (output / f"{Path(result.path).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        count += 1
    return {"images": count, "output": str(output), "protocol": "VisDrone-format predictions; run official evaluator separately"}

