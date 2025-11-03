# train_4070.py
from pathlib import Path
import os, torch
from ultralytics import YOLO

# ----------- 性能/稳定性建议 -----------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # 只用第0张卡
torch.backends.cuda.matmul.allow_tf32 = True    # 支持TF32
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True           # 固定输入尺寸时更快
try:
    torch.set_float32_matmul_precision("high")  # PyTorch>=2.0
except Exception:
    pass

# ----------- 路径配置（这里改成官方原版yaml）-----------
DATA_YAML = Path(r"GC10dataset/data.yaml").resolve()
MODEL_YAML = "yolov8n.yaml"   # ★ 原版结构
PRETRAIN   = "yolov8n.pt"     # 官方预训练权重

def main():
    assert torch.cuda.is_available(), "未检测到 CUDA，请安装带CUDA的PyTorch或检查驱动。"
    print(f"[INFO] Device name: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Using: cuda:0")

    assert DATA_YAML.exists(), f"data.yaml 不存在: {DATA_YAML}"

    # 构建模型（使用官方原版 yolov8n.yaml）
    model = YOLO(MODEL_YAML)
    # 加载官方预训练权重
    model.load(PRETRAIN)

    # 训练
    model.train(
        data=str(DATA_YAML),
        imgsz=640,
        batch=8,             # 根据显存调节
        workers=4,           # Windows建议 4~8
        cache="disk",
        amp=True,            # 混合精度
        device=0,
        epochs=250,
        patience=50,
        cos_lr=True,
        close_mosaic=20,
        lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5, shear=0.0,
        perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0,
        project="runs/train",
        name="gc10-det_yolov8n_vanilla",
        exist_ok=False,
        save_period=25
    )

    # 验证
    model.val(data=str(DATA_YAML), imgsz=640, split="val", device=0)

if __name__ == "__main__":
    main()
