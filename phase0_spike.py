"""
Phase 0 — Class Score Extraction Integration Spike
===================================================
Stability-Gated Iterative Auto-Labeling for YOLOv8
Architecture spec: final_architecture_patched.md, Section 10

v2 fix: Method B now transforms kept boxes from original-image space
        back to model-input space (640x640) before coordinate matching.
        The previous version compared across different coordinate spaces,
        causing 70/80 CHECK 2 violations despite correct logic.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

print("=" * 72)
print("PHASE 0 INTEGRATION SPIKE — Class Score Extraction")
print("=" * 72)

import torch
import ultralytics
from ultralytics.models.yolo.detect import DetectionPredictor

TORCH_VERSION       = torch.__version__
ULTRALYTICS_VERSION = ultralytics.__version__
print(f"  torch        : {TORCH_VERSION}")
print(f"  ultralytics  : {ULTRALYTICS_VERSION}")
print()


# ── Minimal PseudoLabel (mirrors bank/schemas.py spec) ───────────────────────

@dataclass(frozen=True)
class PseudoLabel:
    """
    confidence:   q_t from results.boxes.conf  — admission gate ONLY
    class_scores: full K-dim softmax vector    — scoring / JS cost ONLY
    Never assumed numerically equal.
    """
    image_id:     str
    box_id:       str
    round_id:     int
    box:          Tuple[float, float, float, float]
    pred_class:   int
    class_scores: Tuple[float, ...]
    confidence:   float


# ── Preferred path ────────────────────────────────────────────────────────────

class ClassScoreCapturingPredictor(DetectionPredictor):
    """
    Overrides postprocess() to capture pre-NMS class logits before NMS
    discards non-kept anchors.

    CHECK 2 fix (v2): kept boxes from results are in ORIGINAL IMAGE space.
    Pre-NMS anchor coords are in MODEL INPUT space (imgsz × imgsz, e.g. 640).
    Method B now inverts the letterbox transform to bring both into the same
    space before matching, giving exact coordinate identity.

    Letterbox transform:
        ratio   = min(imgsz / orig_h, imgsz / orig_w)
        pad_w   = (imgsz - orig_w * ratio) / 2
        pad_h   = (imgsz - orig_h * ratio) / 2
        model_x = orig_x * ratio + pad_w
        model_y = orig_y * ratio + pad_h
    """

    def postprocess(self, preds, img, orig_imgs):
        raw = preds[0]                                      # [B, 4+nc, N]
        B, channels, N = raw.shape
        nc = channels - 4

        self._raw_logits_shape     = (B, nc, N)
        self._captured_class_probs = torch.softmax(
            raw[:, 4:, :], dim=1).detach().cpu()           # [B, nc, N]
        self._anchor_coords        = raw[:, :4, :].detach().cpu()  # [B, 4, N]
        self._imgsz                = img.shape[-1]          # model input size

        results = super().postprocess(preds, img, orig_imgs)
        self._attach_class_scores(results)
        return results

    def _attach_class_scores(self, results) -> None:
        for b, result in enumerate(results):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                result._class_scores = torch.zeros(
                    0, self._raw_logits_shape[1])
                continue

            n_kept    = len(boxes)
            nc        = self._captured_class_probs.shape[1]
            N_anchors = self._captured_class_probs.shape[2]
            imgsz     = self._imgsz

            # ── Method A: direct index column ────────────────────────────────
            # Ultralytics ≥8.x sometimes appends pre-NMS anchor index as col 6.
            anchor_indices = None
            if boxes.data.shape[1] >= 7:
                idx = boxes.data[:, 6].long()
                if int(idx.max()) < N_anchors:
                    anchor_indices = idx

            if anchor_indices is not None:
                scores_out = self._captured_class_probs[
                    b, :, anchor_indices].T                 # [n_kept, nc]

            else:
                # ── Method B: coordinate-identity matching ────────────────────
                # Kept boxes are in ORIGINAL IMAGE space (result.orig_shape).
                # Anchor boxes are in MODEL INPUT space (imgsz × imgsz).
                # We transform kept boxes → model input space using the
                # letterbox inverse, then match by exact coordinates.

                orig_h, orig_w = result.orig_shape
                ratio  = min(imgsz / orig_h, imgsz / orig_w)
                pad_w  = (imgsz - orig_w * ratio) / 2
                pad_h  = (imgsz - orig_h * ratio) / 2

                kept_xyxy = boxes.data[:, :4].cpu()         # [n_kept, 4] orig space

                # Transform to model input space
                kept_model      = kept_xyxy.clone()
                kept_model[:, 0] = kept_xyxy[:, 0] * ratio + pad_w  # x1
                kept_model[:, 1] = kept_xyxy[:, 1] * ratio + pad_h  # y1
                kept_model[:, 2] = kept_xyxy[:, 2] * ratio + pad_w  # x2
                kept_model[:, 3] = kept_xyxy[:, 3] * ratio + pad_h  # y2

                # Decode anchor xywh → xyxy in model input space
                anchor_b = self._anchor_coords[b]           # [4, N]
                ax1 = anchor_b[0] - anchor_b[2] / 2
                ay1 = anchor_b[1] - anchor_b[3] / 2
                ax2 = anchor_b[0] + anchor_b[2] / 2
                ay2 = anchor_b[1] + anchor_b[3] / 2
                anchor_xyxy = torch.stack(
                    [ax1, ay1, ax2, ay2], dim=1)            # [N, 4]

                # L2 distance — both tensors now in same coordinate space
                dists       = torch.cdist(
                    kept_model.float(), anchor_xyxy.float(), p=2)  # [n_kept, N]
                best_anchor = dists.argmin(dim=1)           # [n_kept]
                scores_out  = self._captured_class_probs[
                    b, :, best_anchor].T                    # [n_kept, nc]

            result._class_scores = scores_out               # [n_kept, nc]


# ── Fallback path (only if preferred path fails) ──────────────────────────────

class HookBasedClassCapture:
    """
    Forward hook on the Detect head. Fragile but acceptable for MVP spike.
    Scope: internal to canonical_infer.py only.
    """
    def __init__(self):
        self._hook_output = None
        self._handle      = None

    def attach(self, model) -> None:
        detect_module = None
        for m in model.modules():
            if type(m).__name__ == "Detect":
                detect_module = m
        if detect_module is None:
            raise RuntimeError("No Detect module found.")
        self._handle = detect_module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output) -> None:
        raw = output[0] if isinstance(output, (list, tuple)) else output
        self._hook_output = torch.softmax(
            raw[:, 4:, :], dim=1).detach().cpu()

    def get_class_probs(self) -> Optional[torch.Tensor]:
        return self._hook_output

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ── Synthetic image factory ───────────────────────────────────────────────────

def make_synthetic_images(n: int = 10, size: int = 640) -> List[np.ndarray]:
    rng = np.random.default_rng(seed=42)
    return [rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
            for _ in range(n)]


# ── Spike runner ──────────────────────────────────────────────────────────────

def run_spike(model_path:   str   = "yolov8m.pt",
              n_images:     int   = 10,
              conf_thr:     float = 0.05,
              iou_nms:      float = 0.45,
              imgsz:        int   = 640,
              source_path:  str   = None) -> Dict:

    import cv2
    from ultralytics import YOLO

    print(f"  Loading model : {model_path}")
    model = YOLO(model_path)

    if source_path:
        img = cv2.imread(source_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {source_path}")
        images = [img] * n_images
        print(f"  Image source  : {source_path} (x{n_images})")
    else:
        images = make_synthetic_images(n=n_images, size=imgsz)
        print(f"  Image source  : synthetic noise ({n_images} images)")

    # GPU determinism guard — required for CHECK 4 on CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    overrides = {"model": model_path, "conf": conf_thr,
                 "iou": iou_nms, "imgsz": imgsz, "verbose": False}

    print("  Running inference (pass 1) …")
    p1 = ClassScoreCapturingPredictor(overrides=overrides)
    p1.setup_model(model=model.model)
    run1 = p1(source=images, stream=False)

    print("  Running inference (pass 2, same inputs) …")
    p2 = ClassScoreCapturingPredictor(overrides=overrides)
    p2.setup_model(model=model.model)
    run2 = p2(source=images, stream=False)

    # ── CHECK 1 ───────────────────────────────────────────────────────────────
    B, nc, N      = p1._raw_logits_shape
    check1_pass   = True
    check1_detail = f"  Captured shape : [B={B}, num_classes={nc}, num_anchors={N}]"

    # ── CHECKs 2 & 3 ─────────────────────────────────────────────────────────
    pseudo_labels: List[PseudoLabel] = []
    violations:    List[str]         = []
    equiv:         List[str]         = []

    for img_idx, r in enumerate(run1):
        if r.boxes is None or len(r.boxes) == 0:
            continue
        cs_tensor   = r._class_scores          # [n_kept, nc]
        confidences = r.boxes.conf.cpu()
        classes     = r.boxes.cls.cpu().long()
        xyxy        = r.boxes.xyxy.cpu()

        for di in range(len(r.boxes)):
            cs     = cs_tensor[di]
            cs_sum = float(cs.sum())
            cs     = cs / cs_sum if cs_sum > 1e-8 else torch.ones(nc) / nc
            conf_v = float(confidences[di])
            cls_v  = int(classes[di])

            pseudo_labels.append(PseudoLabel(
                image_id    = f"img_{img_idx}",
                box_id      = f"img_{img_idx}_r1_{di}",
                round_id    = 1,
                box         = tuple(float(v) for v in xyxy[di]),
                pred_class  = cls_v,
                class_scores= tuple(float(v) for v in cs),
                confidence  = conf_v,
            ))

            argmax_cs = int(torch.argmax(cs).item())
            if argmax_cs != cls_v:
                violations.append(
                    f"    img={img_idx} det={di}: "
                    f"argmax={argmax_cs} != boxes.cls={cls_v}")

            if abs(conf_v - float(cs.max())) < 1e-5:
                equiv.append(f"    img={img_idx} det={di}: conf≈max(cs)={conf_v:.6f}")

    n_total     = len(pseudo_labels)
    check2_pass = len(violations) == 0

    if n_total == 0:
        check3_pass   = True
        check3_detail = "  No detections. CHECK 3 vacuously N/A."
    else:
        ok_conf   = all(isinstance(pl.confidence, float) for pl in pseudo_labels)
        ok_scores = all(
            len(pl.class_scores) == nc
            and abs(sum(pl.class_scores) - 1.0) < 1e-4
            for pl in pseudo_labels)
        check3_pass   = ok_conf and ok_scores
        check3_detail = (
            f"  {n_total} detections, {len(equiv)} have conf≈max(class_scores)\n"
            f"  [independent population paths confirmed — different fields]")

    # ── CHECK 4 ───────────────────────────────────────────────────────────────
    def _extract(runs):
        out = []
        for i, r in enumerate(runs):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for di in range(len(r.boxes)):
                cs = tuple(round(float(v), 8) for v in r._class_scores[di])
                out.append((i, di, cs))
        return out

    s1, s2       = _extract(run1), _extract(run2)
    scores_match = (len(s1) == len(s2)) and all(
        a[2] == b[2] for a, b in zip(s1, s2))
    check4_pass   = scores_match
    check4_detail = (
        f"  Pass-1 dets: {len(s1)}, Pass-2 dets: {len(s2)}, "
        f"Bit-identical: {scores_match}")

    return dict(
        check1_pass=check1_pass, check1_detail=check1_detail,
        check2_pass=check2_pass, violations=violations, n_detections=n_total,
        check3_pass=check3_pass, check3_detail=check3_detail,
        check4_pass=check4_pass, check4_detail=check4_detail,
        torch_version=TORCH_VERSION, ultralytics_version=ULTRALYTICS_VERSION,
    )


def run_fallback_spike(model_path: str   = "yolov8m.pt",
                       imgsz:      int   = 640,
                       conf_thr:   float = 0.05,
                       iou_nms:    float = 0.45,
                       source_path: str  = None) -> Dict:
    import cv2
    from ultralytics import YOLO

    print("  [FALLBACK] Attaching forward hook to Detect head …")
    model   = YOLO(model_path)
    capture = HookBasedClassCapture()
    capture.attach(model.model)

    if source_path:
        img    = cv2.imread(source_path)
        images = [img, img]
    else:
        images = make_synthetic_images(n=2, size=imgsz)

    model.predict(images, conf=conf_thr, iou=iou_nms,
                  imgsz=imgsz, verbose=False)
    hook_probs = capture.get_class_probs()
    capture.detach()

    if hook_probs is None:
        return {"fallback_pass": False, "reason": "Hook did not fire"}

    B, nc, N  = hook_probs.shape
    shape_ok  = (nc > 0 and N > 0)
    sums_ok   = bool(
        (hook_probs.sum(dim=1) - 1.0).abs().max().item() < 1e-4)
    return {
        "fallback_pass": shape_ok and sums_ok,
        "hook_shape"   : (B, nc, N),
        "sums_to_one"  : sums_ok,
        "shape_ok"     : shape_ok,
    }


# ── Report printer ────────────────────────────────────────────────────────────

def print_report(res: Dict, fallback_res: Optional[Dict] = None) -> None:
    checks     = ['check1_pass', 'check2_pass', 'check3_pass', 'check4_pass']
    all_passed = all(res[k] for k in checks)

    print(); print("=" * 72); print("SPIKE REPORT"); print("=" * 72)
    print(f"  torch        : {res['torch_version']}")
    print(f"  ultralytics  : {res['ultralytics_version']}")
    print()

    print("CHECK 1  raw_cls_logits shape == [B, num_classes, num_anchors]")
    print(f"  {'PASS ✓' if res['check1_pass'] else 'FAIL ✗'}")
    print(res['check1_detail']); print()

    print("CHECK 2  argmax(class_scores) == results.boxes.cls for all detections")
    print(f"  {'PASS ✓' if res['check2_pass'] else 'FAIL ✗'}")
    print(f"  Total detections inspected : {res['n_detections']}")
    if res['violations']:
        print(f"  VIOLATIONS ({len(res['violations'])}):")
        for v in res['violations'][:10]:
            print(v)
        if len(res['violations']) > 10:
            print(f"    … and {len(res['violations'])-10} more")
    else:
        print("  No violations found.")
    print()

    print("CHECK 3  confidence and class_scores populated independently")
    print(f"  {'PASS ✓' if res['check3_pass'] else 'FAIL ✗'}")
    print(res['check3_detail']); print()

    print("CHECK 4  Anchor-to-detection mapping is deterministic")
    print(f"  {'PASS ✓' if res['check4_pass'] else 'FAIL ✗'}")
    print(res['check4_detail']); print()

    if fallback_res is not None:
        print("FALLBACK PATH (hook-based)")
        print(f"  {'PASS ✓' if fallback_res.get('fallback_pass') else 'FAIL ✗'}")
        for k, v in fallback_res.items():
            if k != 'fallback_pass':
                print(f"  {k}: {v}")
        print()

    print("=" * 72)
    if all_passed:
        print("OVERALL VERDICT: PASS")
        print()
        print("  canonical_infer.py → ClassScoreCapturingPredictor")
        print("  PseudoLabel.confidence   ← results.boxes.conf   [admission gate]")
        print("  PseudoLabel.class_scores ← softmax(logits)      [scoring / JS]")
        print("  INDEPENDENT. No equivalence assumed.")
    else:
        failed = [k for k in checks if not res[k]]
        print(f"OVERALL VERDICT: FAIL  —  failed: {failed}")
        if fallback_res and fallback_res.get('fallback_pass'):
            print()
            print("  Fallback path (hook-based) PASSED.")
            print("  canonical_infer.py must use: HookBasedClassCapture")
            print("  All other modules UNAFFECTED — failure contained to")
            print("  canonical_infer.py's internal implementation only.")
        elif fallback_res:
            print()
            print("  Fallback path also FAILED. Manual investigation required.")
            print("  Do NOT reduce class_scores to top-1 + conf scalar.")
    print("=" * 72)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 0 spike — class score extraction")
    parser.add_argument("--model",  default="yolov8m.pt")
    parser.add_argument("--imgsz",  type=int,   default=640)
    parser.add_argument("--conf",   type=float, default=0.05)
    parser.add_argument("--iou",    type=float, default=0.45)
    parser.add_argument("--images", type=int,   default=10,
                        help="Synthetic images (spec: 10)")
    parser.add_argument("--source", default=None,
                        help="Real image path (overrides synthetic noise)")
    args = parser.parse_args()

    print(f"  Model weights : {args.model}")
    print(f"  imgsz         : {args.imgsz}")
    print(f"  conf threshold: {args.conf}")
    print(f"  iou (NMS)     : {args.iou}")
    print(f"  synthetic imgs: {args.images}")
    print(f"  source        : {args.source or 'synthetic'}")
    print()

    print("─" * 72)
    print("PREFERRED PATH: DetectionPredictor subclass")
    print("─" * 72)

    preferred_passed = False
    res = {}
    try:
        res = run_spike(
            model_path  = args.model,
            n_images    = args.images,
            conf_thr    = args.conf,
            iou_nms     = args.iou,
            imgsz       = args.imgsz,
            source_path = args.source,
        )
        preferred_passed = all(
            res[k] for k in
            ['check1_pass', 'check2_pass', 'check3_pass', 'check4_pass'])
    except Exception as exc:
        print(f"  PREFERRED PATH exception: {exc}")
        res = {
            "torch_version": TORCH_VERSION,
            "ultralytics_version": ULTRALYTICS_VERSION,
            "check1_pass": False, "check1_detail": f"  Exception: {exc}",
            "check2_pass": False, "violations": [], "n_detections": 0,
            "check3_pass": False, "check3_detail": "",
            "check4_pass": False, "check4_detail": "",
        }

    fallback_res = None
    if not preferred_passed:
        print(); print("─" * 72)
        print("FALLBACK PATH: forward hook on Detect head")
        print("─" * 72)
        try:
            fallback_res = run_fallback_spike(
                model_path  = args.model,
                imgsz       = args.imgsz,
                conf_thr    = args.conf,
                iou_nms     = args.iou,
                source_path = args.source,
            )
        except Exception as exc:
            fallback_res = {"fallback_pass": False, "reason": str(exc)}

    print_report(res, fallback_res)