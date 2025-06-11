#!/usr/bin/env python3

"""
KAIST-RGBT train-all-04.txt → train/val 세트 분리 스크립트
  1) labels/*.txt 를 스캔해 클래스별 ‘이미지 개수’ 통계 출력
  2) person(ID 0)·person?(ID 3) 이 존재하는 이미지의 20 %만 val 세트로 사용
"""

import random
from pathlib import Path
from collections import defaultdict

# ──────────────── 사용자 설정 ────────────────
INPUT_LIST      = Path("/home/hyunjun/AUE8088-PA2/datasets/kaist-rgbt/train-all-04.txt")
TRAIN_LIST_OUT  = Path("/home/hyunjun/AUE8088-PA2/datasets/kaist-rgbt/train_set.txt")
VAL_LIST_OUT    = Path("/home/hyunjun/AUE8088-PA2/datasets/kaist-rgbt/val_set.txt")
LABEL_DIR       = Path("/home/hyunjun/AUE8088-PA2/datasets/kaist-rgbt/train/labels")
VAL_RATIO       = 0.20                    # person 이 포함된 이미지의 20 %
PERSON_CLS_IDS  = {0, 3}                  # ‘person’ 으로 간주할 클래스 ID 집합
SEED            = 77                      # 재현 가능성 고정
# ────────────────────────────────────────────

random.seed(SEED)

# ---------- 1) 이미지 경로 로드 ----------
with INPUT_LIST.open() as f:
    img_paths = [line.strip() for line in f if line.strip()]

print(f"총 이미지 수: {len(img_paths):,}")

# ---------- 2) 라벨 스캔하며 통계 및 person 이미지 수집 ----------
class_image_counter = defaultdict(int)   # {cls_id: 이미지 수}
person_imgs, other_imgs = [], []

for img_path in img_paths:
    stem = Path(img_path).stem
    label_path = LABEL_DIR / f"{stem}.txt"
    if not label_path.exists():
        # 라벨이 없으면 통계에서 제외
        continue

    # 해당 이미지에 등장한 클래스 ID 모음
    cls_ids_in_img = set()
    with label_path.open() as lf:
        for line in lf:
            if not line.strip():
                continue
            cls_id = int(float(line.split()[0]))   # 0  x y w h
            cls_ids_in_img.add(cls_id)

    # 이미지별(class-level) 카운트 → 한 이미지에 클래스가 여러 번 나와도 1로 셈
    for cid in cls_ids_in_img:
        class_image_counter[cid] += 1

    # person 포함 여부에 따라 분리
    if cls_ids_in_img & PERSON_CLS_IDS:
        person_imgs.append(img_path)
    else:
        other_imgs.append(img_path)

# ---------- 3) validation 이미지 선정 ----------
val_count = int(len(person_imgs) * VAL_RATIO)
val_imgs  = random.sample(person_imgs, val_count)
train_imgs = other_imgs + [p for p in person_imgs if p not in val_imgs]

# ---------- 4) 결과 저장 ----------
# TRAIN_LIST_OUT.write_text("\n".join(train_imgs) + "\n")
# VAL_LIST_OUT.write_text("\n".join(val_imgs) + "\n")

# ---------- 5) 통계 출력 ----------
print("\n📊 클래스별 ‘이미지 수’ 통계")
for cid in sorted(class_image_counter.keys()):
    name = {0: "person", 1: "cyclist", 2: "people", 3: "person?"}.get(cid, f"class_{cid}")
    print(f"  {cid:>2d} ({name:<8}) : {class_image_counter[cid]:,}")

print(f"\nTrain  세트: {len(train_imgs):,}  (person 이미지 {len(train_imgs)-len(other_imgs):,}, 기타 {len(other_imgs):,})")
print(f"Val    세트: {len(val_imgs):,}  (person 이미지의 {VAL_RATIO*100:.0f} %)")