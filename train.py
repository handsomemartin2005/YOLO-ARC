
from pathlib import Path
import torch
from ultralytics import YOLO
from datetime import datetime

# ----------- 路径配置 -----------
DATA_YAML = Path(r"datasets/data.yaml").resolve()
MODEL_YAML = Path(r"D:\code\yolov8\ultralytics\cfg\models\v8\yolov8n_ard_rgcu_clag.yaml").resolve()
#PRETRAIN   = r"D:\code\yolov8\yolov8n.pt"
PRETRAIN   = r"D:\code\yolov8\runs\train\neu-det_yolov8n_ard_rgcu_clag_20250920-1714 0.736\weights\best.pt"

def main():
    assert torch.cuda.is_available(), "未检测到 CUDA"
    print(f"[INFO] 使用设备: {torch.cuda.get_device_name(0)}")

    model = YOLO(str(MODEL_YAML))
    model.load(PRETRAIN)

    # ====== 生成时间戳 ======
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    exp_name = f"neu-det_yolov8n_ard_rgcu_clag_{timestamp}"

    # ====== 训练 ======
    model.train(
        data=str(DATA_YAML),
        epochs=500,
        imgsz=640,
        batch=64,
        device=0,
        workers=6,
        project=r"D:\code\yolov8\runs\train",
        name=exp_name,
        cache="disk",
        exist_ok=False,

    )

    # ====== 验证 ======
    model.val(
        data=str(DATA_YAML),
        imgsz=640,
        split="val",
        device=0,
        project=r"D:\code\yolov8\runs\val",
        name=exp_name,
        exist_ok=False,
    )

    # ====== 测试 ======
    model.val(
        data=str(DATA_YAML),
        imgsz=640,
        split="test",
        device=0,
        project=r"D:\code\yolov8\runs\test",
        name=exp_name,
        exist_ok=False,
    )

if __name__ == "__main__":
    main()
