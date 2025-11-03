# predict.py
# 用法示例：
#   python predict.py                    # 自动使用最近一次训练的 best.pt，在 datasets/images/val 上推理
#   python predict.py --source path/to/img_or_dir
#   python predict.py --weights runs/train/neu-det_yolov8n3/weights/best.pt

from pathlib import Path
import argparse
import os
from ultralytics import YOLO

def find_latest_best(train_dir: Path, name_prefix="neu-det_yolov8n") -> Path:
    """在 runs/train 下查找最新的 best.pt"""
    if not train_dir.exists():
        raise FileNotFoundError(f"Train dir not found: {train_dir}")
    cands = []
    for d in train_dir.iterdir():
        if d.is_dir() and d.name.startswith(name_prefix):
            best = d / "weights" / "best.pt"
            if best.exists():
                cands.append(best)
    if not cands:
        raise FileNotFoundError(f"No best.pt found under {train_dir} (prefix='{name_prefix}')")
    return max(cands, key=lambda p: p.stat().st_mtime)

def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser("YOLOv8 Neu-DET Predict")
    parser.add_argument("--weights", type=str, default="", help="权重路径；留空则自动找最新 best.pt")
    parser.add_argument("--source", type=str, default=str(base / "datasets" / "images" / "val"),
                        help="图片/文件夹/视频/通配符/摄像头索引")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--name", type=str, default="", help="输出目录名；留空则使用训练目录名")
    parser.add_argument("--save_txt", action="store_true", help="保存 YOLO txt 结果")
    parser.add_argument("--save_conf", action="store_true", help="在 txt 中保存置信度")
    args = parser.parse_args()

    # 选择权重
    if args.weights:
        weights = Path(args.weights).expanduser().resolve()
        if not weights.exists():
            raise FileNotFoundError(f"Weights not found: {weights}")
        run_name = args.name or weights.parent.parent.name  # 训练目录名
    else:
        weights = find_latest_best(base / "runs" / "train", name_prefix="neu-det_yolov8n")
        run_name = args.name or weights.parent.parent.name

    print(f"[INFO] Using weights: {weights}")
    print(f"[INFO] Source       : {args.source}")

    model = YOLO(str(weights))
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        half=True if args.device != "cpu" else False,  # GPU 上用 FP16
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        show_labels=True,
        show_conf=True,
        show_boxes=True,
        line_width=2,
        project=str(base / "runs" / "predict"),
        name=run_name,           # 用训练目录名做子目录
        exist_ok=False           # 自动递增，不覆盖
    )

if __name__ == "__main__":
    # 可选：缓解潜在的显存碎片（预测一般不需要，留着也无害）
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    main()
