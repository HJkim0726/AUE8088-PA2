#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
labels/*.txt 안의 class 0 (pedestrian) 박스 크기를 K-means로 클러스터링
- YOLO 형식: class xc yc w h [extra]
- w, h는 정규화 값(0~1) → 이미지 실제 크기로 환산해 사용
"""

import os, glob
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans   # pip install scikit-learn

# ----------------- 사용자 환경에 맞게 수정 -----------------
LABEL_DIR  = 'datasets/kaist-rgbt/train/labels'   # txt 파일 위치
IMAGE_DIR  = 'datasets/kaist-rgbt/train/images/lwir'   # 대응 이미지 위치
IMG_EXTS   = ('.jpg', '.png', '.jpeg')            # 이미지 확장자 후보
K          = 9                                    # 원하는 군집 수
# ---------------------------------------------------------

box_sizes = []  # (w, h) 목록

for lbl_path in glob.glob(os.path.join(LABEL_DIR, '*.txt')):
    stem = os.path.splitext(os.path.basename(lbl_path))[0]  # 파일명(확장자 제외)

    # 이미지 파일 경로 추정
    img_path = next((os.path.join(IMAGE_DIR, stem + ext)
                     for ext in IMG_EXTS if os.path.isfile(os.path.join(IMAGE_DIR, stem + ext))), None)
    if img_path is None:
        print(f'[WARN] 이미지 없음: {stem}.*');  continue

    # 이미지 크기
    with Image.open(img_path) as im:
        img_w, img_h = im.size

    # 라벨 파일 파싱
    with open(lbl_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:            # 최소 YOLO 필드 (class xc yc w h)
                continue
            cls = int(parts[0])
            if cls != 0:                  # class 0 → pedestrian
                continue
            w_norm, h_norm = map(float, parts[3:5])  # width, height (정규화)
            w_px = w_norm * img_w
            h_px = h_norm * img_h
            box_sizes.append([w_px, h_px])

box_sizes = np.array(box_sizes)
print(f'수집한 box 수: {len(box_sizes)}')

# ----------------- K-means -----------------
kmeans = KMeans(n_clusters=K, random_state=0, n_init=20).fit(box_sizes)
anchors = kmeans.cluster_centers_

# 면적(작→큰) 기준 정렬
anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]

print('\n=== 권장 anchor (pixel 단위) ===')
for i, (w, h) in enumerate(anchors):
    print(f'{i:2d}: ({w:.1f}, {h:.1f})')

# YAML에 넣기 위해 리스트 형태로 출력
flat = [f'{w:.0f},{h:.0f}' for w, h in anchors]
print('\nYAML anchors 블록 예시:')
print('  - [' + ', '.join(flat[:3]) + ']  # P3/8')
print('  - [' + ', '.join(flat[3:6]) + ']  # P4/16')
print('  - [' + ', '.join(flat[6:])  + ']  # P5/32')
