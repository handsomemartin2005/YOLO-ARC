# YOLO-ARC: Attention-Restructured Convolution for Small-Target Defect Detection

A lightweight and high-precision small-object detection model based on the YOLOv8 framework.  
Key components: A2DGLUConv, RGCU, and CLAG.

---

## Model Overview

YOLO-ARC is an enhanced version of YOLOv8 designed for industrial small-target defect detection.  
While keeping the end-to-end detection pipeline and decoupled detection head unchanged,  
the model introduces three structural improvements in the backbone and neck:

---

### A2DGLUConv  
**Attention-based Residual Downsampling with Gated Linear Unit**

- Replaces all stride=2 downsampling layers.  
- Introduces an attention-augmented residual branch to suppress irrelevant features.  
- Integrates a Gated Linear Unit (GLU) to improve channel selectivity.  
- Objective: preserve high-frequency texture and edge details during downsampling and reduce aliasing distortion.

---

### RGCU  
**Risk-Gated Converse Upsample**

- A deconvolution-based upsampling module.  
- Combines frequency-domain reconstruction (Converse2D) with a risk-gated modulation mechanism.  
- Adaptively controls feature restoration strength for smoother upsampling of shallow features.  
- Objective: improve robustness and feature consistency for small-object recovery.

---

### CLAG  
**Cross-Layer Attention Guidance**

- Introduces shallow-layer attention maps into the neck fusion stage.  
- Guides deep semantic feature fusion to suppress background noise.  
- Employs shared cross-layer attention weights through a guidance map.  
- Objective: strengthen small-object responses and improve feature alignment across scales.

---

## Environment Setup

```bash
git clone https://github.com/handsomemartin2005/YOLO-ARC.git
cd YOLO-ARC

conda create -n yolo-arc python=3.10
conda activate yolo-arc

pip install -r requirements.txt
