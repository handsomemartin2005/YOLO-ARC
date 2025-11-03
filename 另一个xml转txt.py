import os
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Set

# === 1) 路径设置（改这里） ===
ROOT = Path(r"C:\Users\ZiduZhang\Desktop\GC10-DET")   # GC10-DET 根目录
SRC_DIR = ROOT / "lable"                              # VOC XML 存放处（原数据即为 lable）
OUT_DIR = ROOT / "labels"                             # YOLO txt 输出目录
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === 2) 类别映射 ===
# 如果你已经确定类别顺序（与训练时一致），在这里给出：
# 例如：CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches", ...]
CLASSES: Optional[List[str]] = None  # 留空则自动从 XML 收集并排序写入 classes.txt

def _get_text(parent, tag, default=None):
    node = parent.find(tag)
    return node.text.strip() if node is not None and node.text is not None else default

def _to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def scan_all_class_names(xml_dir: Path) -> List[str]:
    """从所有 XML 中收集类别名，并按字典序排序。"""
    names: Set[str] = set()
    for p in xml_dir.glob("*.xml"):
        try:
            root = ET.parse(p).getroot()
            for obj in root.findall("object"):
                name = _get_text(obj, "name", "").strip()
                if name:
                    names.add(name)
        except Exception:
            # 跳过损坏文件
            pass
    classes = sorted(names)
    return classes

def write_classes_txt(classes: List[str], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for c in classes:
            f.write(c + "\n")

def voc_to_yolo_single(xml_path: Path, class_to_id: Dict[str, int]) -> int:
    root = ET.parse(xml_path).getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"{xml_path.name} 缺少 <size> 节点")

    W = _to_float(_get_text(size, "width"))
    H = _to_float(_get_text(size, "height"))
    if not W or not H or W <= 0 or H <= 0:
        raise ValueError(f"{xml_path.name} 尺寸非法: width={W}, height={H}")

    lines = []
    warn_clip = 0
    for obj in root.findall("object"):
        name = _get_text(obj, "name", "").strip()
        if name not in class_to_id:
            # 未知类别直接跳过（也可以选择 raise）
            continue
        cid = class_to_id[name]

        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = _to_float(_get_text(box, "xmin"))
        ymin = _to_float(_get_text(box, "ymin"))
        xmax = _to_float(_get_text(box, "xmax"))
        ymax = _to_float(_get_text(box, "ymax"))
        if None in (xmin, ymin, xmax, ymax):
            continue

        # 纠正与裁剪
        x1 = max(0.0, min(xmin, xmax, W))
        y1 = max(0.0, min(ymin, ymax, H))
        x2 = min(W, max(xmin, xmax, 0.0))
        y2 = min(H, max(ymin, ymax, 0.0))
        if (x1, y1, x2, y2) != (xmin, ymin, xmax, ymax):
            warn_clip += 1

        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        if bw <= 0 or bh <= 0:
            continue

        # 中心与归一化
        cx = (x1 + x2) / 2.0 / W
        cy = (y1 + y2) / 2.0 / H
        nw = bw / W
        nh = bh / H

        lines.append(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    # 写出（空文件也写，方便训练框架对齐）
    out_txt = OUT_DIR / f"{xml_path.stem}.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    if warn_clip:
        print(f"[INFO] {xml_path.name}: {warn_clip} 个框发生边界裁剪")
    return len(lines)

def main():
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"找不到标注目录：{SRC_DIR}")

    # 准备类别映射
    if CLASSES is None:
        classes = scan_all_class_names(SRC_DIR)
        if not classes:
            raise RuntimeError("未在 XML 中扫描到任何类别名。")
        write_classes_txt(classes, ROOT / "classes.txt")
        print(f"[OK] 自动收集到 {len(classes)} 个类别，已写入 {ROOT/'classes.txt'}：{classes}")
    else:
        classes = CLASSES
        write_classes_txt(classes, ROOT / "classes.txt")
        print(f"[OK] 使用手动类别列表（{len(classes)}）并写入 {ROOT/'classes.txt'}：{classes}")

    class_to_id = {c: i for i, c in enumerate(classes)}

    total_xml = 0
    total_boxes = 0
    for xml_file in SRC_DIR.glob("*.xml"):
        total_xml += 1
        try:
            n = voc_to_yolo_single(xml_file, class_to_id)
            total_boxes += n
        except Exception as e:
            print(f"[ERROR] 处理 {xml_file.name} 失败：{e}")

    print(f"\n完成：处理 {total_xml} 个 XML，写出 {total_boxes} 个 bbox。YOLO 标签保存在：{OUT_DIR}")
    print("类别映射见：", ROOT / "classes.txt")

if __name__ == "__main__":
    main()
