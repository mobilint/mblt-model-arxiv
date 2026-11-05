[English](ARCHITECTURE.md)

# Architecture

이 repository에는 두 종류의 폴더가 있습니다. 각 폴더는 완결되어 있어서, 의존하는 폴더와 함께
복사하면 문서대로 실행됩니다.

```text
models/<task>/<Model>/   Model 하나: export, calibration, compile, evaluation, inference
datasets/<Dataset>/      Dataset 하나: 준비와 채점
```

## 공개 interface

- 폴더의 공개 interface는 진입 script와 그 JSON 입력·출력입니다. 모든 진입 script는
  `--help`를 지원합니다.
  - Model: `export.py`, `prepare_calibration.py`, `compile.py`, `evaluate.py`, `inference.py`, `inference_sync.py`, `inference_async.py`
  - Dataset: `prepare.py`, `score.py` (`samples`와 `score`)
- 나머지 `.py` 파일은 그 폴더의 helper입니다. 그 폴더의 script만 import하며, 폴더 밖에서는
  import하지 않습니다.
- `.json`, `.yaml`, `.txt` 파일은 그 폴더의 설정과 데이터이며, 그 폴더의 script가 읽습니다.

## 의존

- Model 폴더끼리, Dataset 폴더끼리는 서로 의존하지 않습니다.
- Model은 자기가 가리키는 Dataset에만 의존합니다: `calibration_spec.json`은 calibration에 쓸
  이미지를 준비하는 Dataset을, `model_contract.json`은 evaluation Dataset과 그 폴더
  (`evaluation.dataset_code`)를 가리킵니다.
- Model은 Dataset의 `score.py`를 실행하고 그 JSON 출력을 읽는 방식으로만 Dataset을 씁니다.
  Dataset 코드를 import하지 않으며, Dataset은 Model을 쓰지 않습니다.
- 그 밖에 폴더에 필요한 것은 사용자가 넘깁니다: 데이터는 경로 인자로, 공개 Python package와
  Mobilint SDK는 설치된 의존성으로 넘깁니다. 공개 package는 폴더의 `requirements.txt`에
  있습니다.

## 데이터 계약

- `score.py samples --task TASK --data-path DIR`은 채점 순서대로
  `{"samples": [{"image": ..., "image_id": ...}, ...]}`를 출력합니다.
- prediction artifact는 `{"task": ..., "image_ids": [...], "records": [...]}`이며, id와
  record field는 Dataset의 `dataloader.py`와 `evaluator.py`가 정합니다.
- `score.py score --data-path DIR --artifact FILE`은 결과
  `{"primary": {...}, "secondary": {...}, "sample_count": N}`을 출력합니다.

## 변경

이 repository는 Mobilint 내부 repository에서 생성되며 손으로 편집하지 않습니다. 문제는 pull
request 대신 issue로 알려 주세요.
