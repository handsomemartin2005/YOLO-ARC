# compare_models_full.py
import os
import time
import csv
from pathlib import Path
import torch
from ultralytics import YOLO

# ============== 路径配置（按需修改） ==============
YAML_VANILLA  = r"D:\code\yolov8\ultralytics\cfg\models\v8\yolov8.yaml"          # 原版
# YAML_ARD      = r"D:\code\yolov8\ultralytics\cfg\models\v8\yolov8n_ard.yaml"      # 只 ARD
# YAML_RGCU     = r"D:\code\yolov8\ultralytics\cfg\models\v8\yolov8n_rgcu.yaml"     # 只 RGCU
# YAML_ARD_RGCU = r"D:\code\yolov8\ultralytics\cfg\models\v8\yolov8n_ard_rgcu.yaml" # ARD+RGCU
# YAML_CLAG       = r"D:\code\yolov8\ultralytics\cfg\models\v8\yolov8n_clag.yaml"   # 只 CLAG
# YAML_ARD_CLAG   = r"D:\code\yolov8\ultralytics\cfg\models\v8\yolov8n_ard_clag.yaml"      # ARD + CLAG
# YAML_RGCU_CLAG  = r"D:\code\yolov8\ultralytics\cfg\models\v8\yolov8n_rgcu_clag.yaml"     # RGCU + CLAG
# YAML_ARD_RGCU_CLAG = r"D:\code\yolov8\ultralytics\cfg\models\v8\yolov8n_ard_rgcu_clag.yaml" # ARD+RGCU+CLAG

IMGSZ   = 640
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP  = 10      # 预热次数
ITERS   = 50      # 统计次数

RESULTS = []  # 保存结果

# ============== 性能开关（4070 建议） ==============
if DEVICE == "cuda":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def load_model_from_yaml(yaml_path: str):
    p = Path(yaml_path)
    assert p.exists(), f"找不到模型yaml: {yaml_path}"
    y = YOLO(str(p))
    m = y.model
    m.eval().to(DEVICE)
    return y, m


def thop_profile(m: torch.nn.Module, imgsz=640):
    try:
        from thop import profile
    except Exception:
        raise RuntimeError("未安装 thop，请先执行: pip install thop")
    x = torch.randn(1, 3, imgsz, imgsz, device=DEVICE)
    try:
        m.fuse()
    except Exception:
        pass
    with torch.no_grad():
        flops, params = profile(m, inputs=(x,), verbose=False)
    return flops, params


@torch.inference_mode()
def latency(m: torch.nn.Module, imgsz=640, amp=True, warmup=10, iters=50):
    x = torch.randn(1, 3, imgsz, imgsz, device=DEVICE)
    # 预热
    for _ in range(warmup):
        with torch.cuda.amp.autocast(enabled=(amp and DEVICE=="cuda")):
            _ = m(x)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        with torch.cuda.amp.autocast(enabled=(amp and DEVICE=="cuda")):
            _ = m(x)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / iters * 1000.0
    return dt


def show(name, yaml_path):
    print(f"\n===== {name} =====")
    y, m = load_model_from_yaml(yaml_path)

    # Params / FLOPs
    try:
        flops, params = thop_profile(m, imgsz=IMGSZ)
        params_m = params / 1e6
        flops_g = flops / 1e9
    except Exception as e:
        print(f"[THOP] 统计失败：{e}")
        params_m, flops_g = 0, 0

    # Latency → FPS
    dt = latency(m, imgsz=IMGSZ, amp=True, warmup=WARMUP, iters=ITERS)
    fps = 1000.0 / dt if dt > 0 else 0

    # Model Size
    model_size = 0
    yaml_p = Path(yaml_path)
    pt_path = yaml_p.with_suffix(".pt")
    if pt_path.exists():
        model_size = os.path.getsize(pt_path) / 1024 / 1024

    RESULTS.append([name, f"{params_m:.3f}", f"{flops_g:.2f}", f"{fps:.1f}", f"{model_size:.1f}"])
    print(f"Params: {params_m:.3f} M")
    print(f"FLOPs : {flops_g:.2f} G")
    print(f"FPS   : {fps:.1f}")
    print(f"Size  : {model_size:.1f} MB")


if __name__ == "__main__":
    print(f"[INFO] Device = {DEVICE}  {torch.cuda.get_device_name(0) if DEVICE=='cuda' else ''}")
    show("YOLOv8n (Vanilla)", YAML_VANILLA)
    show("YOLOv8n + ARD", YAML_ARD)
    show("YOLOv8n + RGCU", YAML_RGCU)
    show("YOLOv8n + CLAG", YAML_CLAG)
    show("YOLOv8n + ARD+RGCU", YAML_ARD_RGCU)
    show("YOLOv8n + ARD+CLAG", YAML_ARD_CLAG)
    show("YOLOv8n + RGCU+CLAG", YAML_RGCU_CLAG)
    show("YOLOv8n + ARD+RGCU+CLAG", YAML_ARD_RGCU_CLAG)

    # 保存 CSV
    header = ["Model", "Params (M)", "GFLOPs (G)", "FPS", "Model Size (MB)"]
    with open("compare_models_full.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(RESULTS)
    print("\n已保存 compare_models_full.csv")

