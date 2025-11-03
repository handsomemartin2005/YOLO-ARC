from pathlib import Path
import os, gc, torch
from ultralytics import YOLO
from datetime import datetime

# ===== 禁用 TensorBoard 回调（避免 on_train_start 画计算图导致 OOM） =====
os.environ["ULTRALYTICS_TENSORBOARD"] = "False"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 可选：减少 TF 日志干扰

# ===== 路径配置 =====
DATA_YAML = Path(r"GC10dataset/data.yaml").resolve()
YAML_DIR  = Path(r"D:\code\yolov8\ultralytics\cfg\models\v8").resolve()

# 预训练权重：放在 YAML_DIR 下，也可以换成你自己的 best.pt
# 如果要从零训练，可以设为 "" 或注释掉加载
PRETRAIN  = str((YAML_DIR / "yolov8n.pt").resolve())

# 要跑的 YAML 文件（已在该目录中）
MODEL_YAMLS = [



    "yolov8n_ard_rgcu_clag.yaml",
    "yolov8n_ard_rgcu.yaml",#√
    "yolov8n_clag.yaml",#√
    "yolov8n_rgcu_clag.yaml",
"yolov8n_ard_clag.yaml",
"yolov8.yaml",
"yolov8n_ard.yaml",#√
    "yolov8n_rgcu.yaml",  # 太慢

]

# 每个 YAML 跑两次
RUNS_PER_MODEL = 1

# ===== 训练/验证公共参数 =====
COMMON_TRAIN_KW = dict(
    data=str(DATA_YAML),
    epochs=300,
    imgsz=640,
    batch=64,
    device=0,
    workers=4,
    cache="disk",
    exist_ok=False,
)
COMMON_VAL_KW = dict(
    data=str(DATA_YAML),
    imgsz=640,
    device=0,
    exist_ok=False,
    split="val",   # 只做验证；不要 test
)

def run_once(model_yaml: Path, run_idx: int):
    """执行一次训练 + 验证，并在结束后释放显存"""
    model = YOLO(str(model_yaml))
    if PRETRAIN and Path(PRETRAIN).exists():
        model.load(PRETRAIN)

    tag = model_yaml.stem
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    exp_name = f"gc10-det_{tag}_{timestamp}_run{run_idx}"

    # --- train ---
    model.train(
        project=str(Path(r"D:\code\yolov8\runs\train_GC10")),
        name=exp_name,
        **COMMON_TRAIN_KW,
    )

    # --- val ---
    model.val(
        project=str(Path(r"D:\code\yolov8\runs\val_GC10")),
        name=exp_name,
        **COMMON_VAL_KW,
    )

    # === 主动释放显存 ===
    del model
    torch.cuda.empty_cache()
    gc.collect()
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

def main():
    assert torch.cuda.is_available(), "未检测到 CUDA"
    print(f"[INFO] 使用设备: {torch.cuda.get_device_name(0)}")

    for yaml_name in MODEL_YAMLS:
        model_yaml = YAML_DIR / yaml_name
        assert model_yaml.exists(), f"未找到: {model_yaml}"
        for i in range(1, RUNS_PER_MODEL + 1):
            print(f"\n[INFO] >>> 开始: {yaml_name} (第 {i}/{RUNS_PER_MODEL} 次)")
            run_once(model_yaml, i)

if __name__ == "__main__":
    main()
