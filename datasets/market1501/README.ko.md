[English](README.md)

# market1501

market1501 검증 데이터를 내려받아 calibration과 평가에 쓸 수 있게 정리하고, 예측 결과를
채점하는 코드입니다.

## 준비

`prepare.py`는 기본 다운로드에 명령줄 도구 `uv` (`uv`)를 실행합니다. `requirements.txt`를 설치하면 함께 설치됩니다.

```bash
python -m pip install -r requirements.txt
python prepare.py --help
python prepare.py --dataset-dir /path/to/market1501
```

`eval/`에는 전체 검증 데이터가, `calib/`에는 calibration 이미지가 만들어집니다. 데이터셋
자체는 포함되어 있지 않습니다. 제공처의 이용 조건을 확인한 뒤 내려받거나, 이미 가지고 있는
압축 파일이나 디렉터리를 `--help`에 나온 source 옵션으로 지정하세요.

## 채점

| 파일 | 역할 |
| --- | --- |
| `prepare.py` | 다운로드, 정리, calibration 이미지 선택 |
| `dataloader.py` | 준비된 검증 샘플을 채점 순서대로 읽기 |
| `evaluator.py` | 정답, 레이블 대응, 평가 지표 |
| `score.py` | 검증 샘플 목록 출력(`samples`)과 예측 결과 채점(`score`) |

```bash
python score.py score --data-path /path/to/market1501/eval --artifact predictions.json
```

예측 결과 파일은 `task`, 채점할 `image_ids`, `records`를 담은 JSON입니다. id의 의미와
각 record의 필드는 `dataloader.py`와 `evaluator.py`가 정합니다.
