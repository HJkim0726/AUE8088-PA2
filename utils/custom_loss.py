# # YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
# """Loss functions (pedestrian‑centric, RGB‑T) with optional regression box loss.

# 업그레이드 내용
# ----------------
# 1. **Ambiguous 라벨 무시**: `ignore_cls_ids` 하이퍼파라미터에 명시된 클래스는 cls loss를
#    계산하지 않고 obj loss만 계산합니다. 기본 person? → `[2]`.
# 2. **Crowd(people) 가중치**: `crowd_cls_ids`에 명시된 클래스에 `crowd_cls_weight`를 곱해
#    cls loss를 낮춥니다. 기본 people → `[1]`, weight `0.5`.
# 3. **클래스 가중치**: `class_weights`(list, len == nc) 지정 시 BCE loss에 곱해 불균형 보정.
# 4. **Quality Focal Loss(QFL)** 옵션: `use_qfl=True`일 때 obj / cls 모두 QFL로 교체.
# 5. **Box regression loss**: `box_loss_type`을 `'reg'` 또는 `'both'`로 설정하면
#    **IoU 기반 손실** 대신/와 함께 **회귀형 손실**(L1, Smooth‑L1, L2) 사용이 가능.
#    * `reg_loss_type`: `'l1'`, `'smooth_l1'`(기본), `'l2'`
#    * `reg_loss_weight`: `'both'` 모드에서 회귀 손실 가중치.
# 6. **타입 힌트 제거 & PyTorch 1.7 호환**: typing 미사용.
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from utils.metrics import bbox_iou
# from utils.torch_utils import de_parallel

# # -----------------------------------------------------------------------------
# # Utilities
# # -----------------------------------------------------------------------------

# def smooth_BCE(eps=0.1):
#     """label smoothing"""
#     return 1.0 - 0.5 * eps, 0.5 * eps


# class BCEBlurWithLogitsLoss(nn.Module):
#     """BCEwithLogitsLoss() with reduced missing‑label effects"""

#     def __init__(self, alpha=0.05):
#         super(BCEBlurWithLogitsLoss, self).__init__()
#         self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")
#         self.alpha = alpha

#     def forward(self, pred, true):
#         loss = self.loss_fcn(pred, true)
#         pred_prob = torch.sigmoid(pred)
#         dx = pred_prob - true
#         alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
#         return (loss * alpha_factor).mean()


# class FocalLoss(nn.Module):
#     """Generic focal loss wrapper"""

#     def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
#         super(FocalLoss, self).__init__()
#         self.loss_fcn = loss_fcn  # must be BCEWithLogitsLoss
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = loss_fcn.reduction
#         self.loss_fcn.reduction = "none"

#     def forward(self, pred, true):
#         loss = self.loss_fcn(pred, true)
#         pred_prob = torch.sigmoid(pred)
#         p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
#         alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
#         modulating_factor = (1.0 - p_t) ** self.gamma
#         loss = loss * alpha_factor * modulating_factor
#         if self.reduction == "mean":
#             return loss.mean()
#         if self.reduction == "sum":
#             return loss.sum()
#         return loss


# class QFocalLoss(nn.Module):
#     """Quality Focal Loss (for objectness & classification)"""

#     def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
#         super(QFocalLoss, self).__init__()
#         self.loss_fcn = loss_fcn  # BCEWithLogitsLoss
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = loss_fcn.reduction
#         self.loss_fcn.reduction = "none"

#     def forward(self, pred, true):
#         loss = self.loss_fcn(pred, true)
#         pred_prob = torch.sigmoid(pred)
#         alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
#         modulating_factor = torch.abs(true - pred_prob) ** self.gamma
#         loss = loss * alpha_factor * modulating_factor
#         if self.reduction == "mean":
#             return loss.mean()
#         if self.reduction == "sum":
#             return loss.sum()
#         return loss


# # -----------------------------------------------------------------------------
# # ComputeLoss (pedestrian‑centric)
# # -----------------------------------------------------------------------------


# class ComputeLoss(object):
#     """YOLOv5 loss with pedestrian‑specific tweaks & optional regression box loss"""

#     sort_obj_iou = False

#     def __init__(self, model, autobalance=False):
#         device = next(model.parameters()).device
#         h = model.hyp  # hyper‑parameters dict

#         # ------------------------------------------------------------------
#         # Hyper‑parameters (추가 항목 포함)
#         # ------------------------------------------------------------------
#         self.ignore_cls_ids = set(h.get("ignore_cls_ids", [2]))  # person?
#         self.crowd_cls_ids = set(h.get("crowd_cls_ids", [1]))    # people
#         self.crowd_cls_weight = h.get("crowd_cls_weight", 0.5)
#         self.class_weights = h.get("class_weights", None)  # list length == nc
#         self.use_qfl = bool(h.get("use_qfl", 0))

#         # **Box loss mode** -------------------------------------------------
#         # 'iou' | 'reg' | 'both'
#         self.box_loss_type = h.get("box_loss_type", "both")
#         self.reg_loss_type = h.get("reg_loss_type", "smooth_l1")
#         self.reg_loss_weight = h.get("reg_loss_weight", 1.0)

#         # Criteria ----------------------------------------------------------
#         BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
#         BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))
#         if self.use_qfl:
#             BCEcls = QFocalLoss(BCEcls, gamma=h.get("fl_gamma", 1.5))
#             BCEobj = QFocalLoss(BCEobj, gamma=h.get("fl_gamma", 1.5))

#         # label smoothing ---------------------------------------------------
#         self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))

#         # focal option ------------------------------------------------------
#         g = h.get("fl_gamma", 0.0)
#         if g > 0 and not self.use_qfl:
#             BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

#         # Detect layer ------------------------------------------------------
#         m = de_parallel(model).model[-1]  # Detect()
#         self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])
#         self.ssi = list(m.stride).index(16) if autobalance else 0  # stride16 index
#         self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
#         self.na, self.nc, self.nl = m.na, m.nc, m.nl
#         self.anchors, self.device = m.anchors, device

#         if self.class_weights is not None:
#             self.class_weights = torch.tensor(self.class_weights, device=device).float()
#             if len(self.class_weights) != self.nc:
#                 raise ValueError("class_weights 길이가 nc와 일치하지 않습니다.")

#     # ---------------------------------------------------------------------
#     # Regression loss helper
#     # ---------------------------------------------------------------------
#     def _reg_loss(self, pbox, tbox):
#         if self.reg_loss_type == "l1":
#             return torch.abs(pbox - tbox).mean()
#         if self.reg_loss_type == "l2":
#             return torch.pow(pbox - tbox, 2).mean()
#         # default smooth‑l1
#         return F.smooth_l1_loss(pbox, tbox, reduction="mean")

#     # ---------------------------------------------------------------------
#     # Call
#     # ---------------------------------------------------------------------
#     def __call__(self, p, targets):
#         lcls = torch.zeros(1, device=self.device)
#         lbox = torch.zeros(1, device=self.device)
#         lobj = torch.zeros(1, device=self.device)
#         tcls, tbox, indices, anchors = self.build_targets(p, targets)

#         # --------------------------------------------------------------
#         # layer‑wise loss
#         # --------------------------------------------------------------
#         for i, pi in enumerate(p):
#             b, a, gj, gi = indices[i]
#             tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)

#             n = b.shape[0]
#             if n:
#                 pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)
#                 pxy = pxy.sigmoid() * 2.0 - 0.5
#                 pwh = (pwh.sigmoid() * 2.0) ** 2 * anchors[i]
#                 pbox = torch.cat((pxy, pwh), 1)

#                 # IoU for objectness & (optionally) box loss
#                 iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()

#                 # ---------------- Box loss -------------------------
#                 if self.box_loss_type == "iou":
#                     lbox += (1.0 - iou).mean()
#                 elif self.box_loss_type == "reg":
#                     lbox += self._reg_loss(pbox, tbox[i])
#                 elif self.box_loss_type == "both":
#                     lbox_iou = (1.0 - iou).mean()
#                     lbox_reg = self._reg_loss(pbox, tbox[i]) * self.reg_loss_weight
#                     lbox += lbox_iou + lbox_reg
#                 else:
#                     raise ValueError("box_loss_type must be 'iou', 'reg', or 'both'")

#                 # IoU‑based objectness target
#                 iou_det = iou.detach().clamp(0).type(tobj.dtype)
#                 if self.sort_obj_iou:
#                     j = iou_det.argsort()
#                     b, a, gj, gi, iou_det = b[j], a[j], gj[j], gi[j], iou_det[j]
#                 if self.gr < 1:
#                     iou_det = (1.0 - self.gr) + self.gr * iou_det

#                 # ignore ambiguous
#                 ignore_mask = (tcls[i] == -1) & (iou_det > self.hyp["iou_t"])
#                 keep = ~ignore_mask
#                 b, a, gj, gi, iou_det, tcls_i = (
#                     b[keep],
#                     a[keep],
#                     gj[keep],
#                     gi[keep],
#                     iou_det[keep],
#                     tcls[i][keep],
#                 )
#                 tobj[b, a, gj, gi] = iou_det

#                 # ---------------- Classification -------------------
#                 if self.nc > 1:
#                     t = torch.full_like(pcls, self.cn, device=self.device)
#                     valid_pos = tcls_i >= 0
#                     if valid_pos.any():
#                         idx_all = torch.arange(tcls_i.shape[0], device=self.device)[valid_pos]
#                         t[idx_all, tcls_i[valid_pos]] = self.cp

#                     cls_loss_raw = self.BCEcls(pcls, t)  # (n, nc) 또는 scalar(QFL)

#                     # class‑wise weight
#                     if self.class_weights is not None:
#                         weights = self.class_weights[tcls_i.clamp(min=0)]
#                         cls_loss_raw = cls_loss_raw.mean(1) * weights
#                     else:
#                         cls_loss_raw = cls_loss_raw.mean(1)

#                     # crowd down‑weight
#                     if self.crowd_cls_ids:
#                         crowd_mask = torch.tensor(
#                             [c.item() in self.crowd_cls_ids for c in tcls_i],
#                             device=self.device,
#                             dtype=torch.bool,
#                         )
#                         cls_loss_raw[crowd_mask] *= self.crowd_cls_weight

#                     lcls += cls_loss_raw.mean()

#             # ---------------- Objectness ---------------------------
#             obj_loss = self.BCEobj(pi[..., 4], tobj)
#             lobj += obj_loss * self.balance[i]
#             if self.autobalance:
#                 self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obj_loss.detach().item()

#         # rebalance --------------------------------------------------
#         if self.autobalance:
#             self.balance = [x / self.balance[self.ssi] for x in self.balance]

#         # scale by hyp ----------------------------------------------
#         lbox *= self.hyp["box"]
#         lobj *= self.hyp["obj"]
#         lcls *= self.hyp["cls"]
#         bs = tobj.shape[0]
#         return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

#     # ---------------------------------------------------------------------
#     # Target builder (동일)
#     # ---------------------------------------------------------------------
#     def build_targets(self, p, targets):
#         na, nt = self.na, targets.shape[0]
#         tcls, tbox, indices, anch = [], [], [], []
#         gain = torch.ones(7, device=self.device)
#         ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
#         targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # add anchor index

#         g = 0.5
#         off = (
#             torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=self.device).float() * g
#         )

#         for i in range(self.nl):
#             anchors, shape = self.anchors[i], p[i].shape
#             gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]

#             t = targets * gain
#             if nt:
#                 r = t[..., 4:6] / anchors[:, None]
#                 j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]
#                 t = t[j]
#                 gxy = t[:, 2:4]
#                 gxi = gain[[2, 3]] - gxy
#                 j1, k1 = ((gxy % 1 < g) & (gxy > 1)).T
#                 l1, m1 = ((gxi % 1 < g) & (gxi > 1)).T
#                 mask = torch.stack((torch.ones_like(j1), j1, k1, l1, m1))
#                 t = t.repeat((5, 1, 1))[mask]
#                 offsets = (torch.zeros_like(gxy)[None] + off[:, None])[mask]
#             else:
#                 t = targets[0]
#                 offsets = 0

#             bc, gxy, gwh, a = t.chunk(4, 1)
#             a, (b, c) = a.long().view(-1), bc.long().T

#             # apply ignore logic BEFORE appending
#             ignore_mask = torch.tensor([cls.item() in self.ignore_cls_ids for cls in c], device=self.device, dtype=torch.bool)
#             c = c.clone()
#             c[ignore_mask] = -1  # mark ambiguous label as -1

#             gij = (gxy - offsets).long()
#             gi, gj = gij.T

#             indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))
#             tbox.append(torch.cat((gxy - gij, gwh), 1))
#             anch.append(anchors[a])
#             tcls.append(c)

#         return tcls, tbox, indices, anch


# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""custom_loss.py (2025‑06‑08)

Pedestrian‑centric YOLOv5 loss with:
* Ambiguous‑label handling & crowd down‑weighting
* Optional Quality Focal Loss
* Optional regression box loss (L1 / Smooth‑L1 / L2 or combined with CIoU)

Changelog
---------
2025‑06‑08 PM · Fix build_targets returning None (append `tcls` & final return)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def smooth_BCE(eps=0.1):
    """Return smoothed positive/negative targets for BCE"""
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        p = torch.sigmoid(pred)
        p_t = true * p + (1 - true) * (1 - p)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        mod_factor = (1.0 - p_t) ** self.gamma
        loss = loss * alpha_factor * mod_factor
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class QFocalLoss(FocalLoss):
    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        p = torch.sigmoid(pred)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        mod_factor = torch.abs(true - p) ** self.gamma
        loss = loss * alpha_factor * mod_factor
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# -----------------------------------------------------------------------------
# ComputeLoss
# -----------------------------------------------------------------------------


class ComputeLoss:
    sort_obj_iou = False

    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device
        hyp = model.hyp
        self.device = device
        self.ignore_cls_ids = set(hyp.get("ignore_cls_ids", [2]))
        self.crowd_cls_ids = set(hyp.get("crowd_cls_ids", [1]))
        self.crowd_cls_weight = hyp.get("crowd_cls_weight", 0.5)
        self.class_weights = hyp.get("class_weights")
        self.use_qfl = bool(hyp.get("use_qfl", 0))
        self.box_loss_type = hyp.get("box_loss_type", "both")
        self.reg_loss_type = hyp.get("reg_loss_type", "smooth_l1")
        self.reg_loss_weight = hyp.get("reg_loss_weight", 1.0)

        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp["cls_pw"]], device=device), reduction="none")
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp["obj_pw"]], device=device), reduction="none")
        if self.use_qfl:
            BCEcls = QFocalLoss(BCEcls, gamma=hyp.get("fl_gamma", 1.5))
            BCEobj = QFocalLoss(BCEobj, gamma=hyp.get("fl_gamma", 1.5))
        else:
            g = hyp.get("fl_gamma", 0.0)
            if g > 0:
                BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
        self.BCEcls, self.BCEobj = BCEcls, BCEobj
        self.cp, self.cn = smooth_BCE(hyp.get("label_smoothing", 0.0))

        m = de_parallel(model).model[-1]
        self.na, self.nc, self.nl = m.na, m.nc, m.nl
        self.anchors = m.anchors
        self.balance = {3: [4.0, 1.0, 0.4]}.get(self.nl, [4.0, 1.0, 0.25, 0.06, 0.02])
        self.ssi = list(m.stride).index(16) if autobalance else 0
        self.hyp = hyp
        self.autobalance = autobalance
        if self.class_weights is not None:
            self.class_weights = torch.tensor(self.class_weights, device=device).float()

    def _reg_loss(self, pbox, tbox):
        if self.reg_loss_type == "l1":
            return torch.abs(pbox - tbox).mean()
        if self.reg_loss_type == "l2":
            return torch.pow(pbox - tbox, 2).mean()
        return F.smooth_l1_loss(pbox, tbox, reduction="mean")

    def __call__(self, preds, targets):
        lcls = torch.zeros(1, device=self.device)
        lbox = torch.zeros(1, device=self.device)
        lobj = torch.zeros(1, device=self.device)
        tcls, tbox, indices, anchors = self.build_targets(preds, targets)
        for i, pi in enumerate(preds):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)
            if b.shape[0]:
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)
                pxy = pxy.sigmoid() * 2.0 - 0.5
                pwh = (pwh.sigmoid() * 2.0) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()
                if self.box_loss_type == "iou":
                    lbox += (1.0 - iou).mean()
                elif self.box_loss_type == "reg":
                    lbox += self._reg_loss(pbox, tbox[i])
                else:
                    lbox += (1.0 - iou).mean() + self._reg_loss(pbox, tbox[i]) * self.reg_loss_weight
                iou_det = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou_det.argsort()
                    b, a, gj, gi, iou_det = b[j], a[j], gj[j], gi[j], iou_det[j]
                tobj[b, a, gj, gi] = iou_det
                if self.nc > 1:
                    t = torch.full_like(pcls, self.cn)
                    pos = tcls[i] >= 0
                    if pos.any():
                        idx = torch.arange(tcls[i].shape[0], device=self.device)[pos]
                        t[idx, tcls[i][pos]] = self.cp
                    cls_loss = self.BCEcls(pcls, t).mean(1)
                    lcls += cls_loss.mean()
            lobj_layer = self.BCEobj(pi[..., 4], tobj).mean()
            lobj += lobj_layer * self.balance[i]
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / lobj_layer.detach().item()
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        return (lbox + lobj + lcls) * targets.shape[0], torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, preds, targets):
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)
        off = torch.tensor([[0,0],[1,0],[0,1],[-1,0],[0,-1]], device=self.device).float() * 0.5
        for i in range(self.nl):
            anchors, shape = self.anchors[i], preds[i].shape
            gain[2:6] = torch.tensor(shape)[[3,2,3,2]]
            t = targets * gain
            if nt:
                r = t[...,4:6]/anchors[:,None]
                j = torch.max(r,1/r).max(2)[0] < self.hyp["anchor_t"]
                t = t[j]
                gxy = t[:,2:4]
                gxi = gain[[2,3]] - gxy
                j1,k1 = ((gxy%1<0.5)&(gxy>1)).T
                l1,m1 = ((gxi%1<0.5)&(gxi>1)).T
                masks = (torch.ones_like(j1), j1, k1, l1, m1)
                mask = torch.stack(masks)
                t = t.repeat((5,1,1))[mask]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[mask]
