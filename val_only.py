import argparse
import torch
from pathlib import Path
import yaml

from models.experimental import attempt_load
from utils.general import check_file, check_img_size
from utils.torch_utils import select_device
from utils.dataloaders import create_dataloader
import val as validate  # YOLOv5 내장 평가 모듈


def run_val_only(weights, data, imgsz=640, batch_size=32, device='', save_json=True, rgbt=False):
    # 디바이스 선택
    device = select_device(device)
    
    # 모델 로드
    model = attempt_load(weights, device=device)
    model.half().to(device) if device.type != 'cpu' else model.to(device)

    # 데이터 경로 확인
    data = check_file(data)
    with open(data) as f:
        data_dict = yaml.safe_load(f)

    # 이미지 크기 확인
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(imgsz, gs)

    # Validation dataloader 생성
    val_path = data_dict['val']
    val_loader = create_dataloader(
        val_path,
        imgsz,
        batch_size,
        gs,
        single_cls=False,
        hyp=None,
        cache=None,
        rect=False,
        rank=-1,
        workers=8,
        pad=0.5,
        prefix="val: ",
        rgbt_input=rgbt,
    )[0]

    # 평가 실행 (JSON 저장 포함)
    validate.run(
        data=data_dict,
        batch_size=batch_size,
        imgsz=imgsz,
        model=model,
        half=device.type != 'cpu',
        dataloader=val_loader,
        save_json=save_json,
        plots=False,
        save_dir=Path('val_output'),
        verbose=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='model path (.pt)')
    parser.add_argument('--data', type=str, required=True, help='data config path (.yaml)')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--device', default='', help='cuda device or cpu')
    parser.add_argument('--rgbt', action='store_true', help='for RGB-T model input')
    opt = parser.parse_args()

    run_val_only(
        weights=opt.weights,
        data=opt.data,
        imgsz=opt.imgsz,
        batch_size=opt.batch_size,
        device=opt.device,
        save_json=True,
        rgbt=opt.rgbt
    )
