[English](README.md)

# DenseNet121_IMAGENET1K_V1

- 작업: 이미지 분류
- 대상 장치: `aries-rb`
- 원본 모델: [TorchVision DenseNet121](https://github.com/pytorch/vision) (라이선스 BSD-3-Clause)
- Checkpoint: <https://download.pytorch.org/models/densenet121-a639ec97.pth>

checkpoint와 데이터셋은 포함되어 있지 않으며, 각각 원 제공처의 이용 조건을 따릅니다.

## 기대 결과

| | 결과 | 조건 | Model 다운로드 |
| --- | --- | --- | --- |
| Mobilint 기록, float (ONNX) | Top-1 0.74434, Top-5 0.91974 | ImageNet-1k 검증 50,000장, ONNX | — |
| 이 폴더로 재현, NPU (MXQ) | Top-1 0.74350, Top-5 0.91838 | qbcompiler 1.4.0, qbruntime 1.5.0 | [mxq ↓](https://huggingface.co/mobilint/DenseNet121/resolve/beed883be2f69165f37a875c358104185dab73f5/aries-rb/densenet121_IMAGENET1K_V1.mxq?download=true) |

MXQ SHA-256: `34c0ed9c62f7f5d9d732bef6a03163c37ef1b48a0d58952dd1621c486844b81c` (파일 `densenet121_IMAGENET1K_V1.mxq`). 내려받은 뒤 `sha256sum densenet121_IMAGENET1K_V1.mxq`으로 확인하세요. 값이 정확히 같아야 합니다. 링크는 Hugging Face repository `mobilint/DenseNet121`의 revision `beed883be2f69165f37a875c358104185dab73f5`에 고정되어 있습니다.

`evaluate.py`가 비교하는 Mobilint NPU 기준값(`model_contract.json`에 저장됨)은 실행 image digest와 입력 이력이 없는 이전 기록(legacy)입니다. 내부 build로 측정했으며 실행 image는 `mobilint/qbcompiler:1.2-cuda12.8.1-ubuntu22.04`입니다. 재현 결과가 그 기록과 허용 오차 안에서 일치하는지 확인했습니다.

## 재현: compile부터 평가까지

Python 3.10, Mobilint SDK의 `qbcompiler`, `qbruntime`, 연결된 NPU가 필요합니다. qbcompiler 1.4.0에 맞추려면 tag가 `1.4-`로 시작하는 `mobilint/qbcompiler` image를 쓰고, GPU에서 compile하면 `-cuda...` tag를, CPU에서 compile하면 `-cpu-...` tag를 고르세요([게시된 tag 목록](https://hub.docker.com/r/mobilint/qbcompiler/tags)). 그 prefix와 요청 실행 방식을 모두 만족하는 tag가 없으면 같은 실행 방식의 가장 최근 게시 tag를 쓰되, qbcompiler wheel은 위 버전을 그대로 설치하고 실제 사용한 image를 결과와 함께 기록하세요. 목록 조회가 성공하여 tag 부재를 확인한 경우에만 fallback하며, 네트워크나 registry 오류에서는 중단하세요. Dataset
코드는 이 repository의 [`datasets/imagenet`](../../../datasets/imagenet/)에 있습니다.

```bash
python -m pip install -r requirements.txt
python ../../../datasets/imagenet/prepare.py --dataset-dir /path/to/imagenet
python compile.py --preflight
python export.py
python prepare_calibration.py --source /path/to/imagenet/eval
python compile.py
python evaluate.py --data-path /path/to/imagenet/eval
```

기본적으로 aries와 aries-rb는 Single, Multi, Global4, Global8 실행 계획을 모두 컴파일합니다. 단일 계획만 필요한 artifact를 만들려면
`--inference-scheme single`을 지정하세요.

`qbcompiler`는 자체 torch 버전을 고정합니다. `export.py`에 더 새로운 torch가 필요하면 export만
별도 환경에서 실행하세요. `evaluate.py`는 결과를 `artifacts/evaluation/result.json`에 쓰고
차이를 보여 줍니다. `--framework mxq`(기본값)는 `model_contract.json`에 저장된 Mobilint의
내부 NPU 기록값(게시된 재현 결과가 명시된 허용 오차 안에서 수용된 기준값)과 비교하며,
`--framework onnx`는 위 float 결과와 비교합니다. `--limit N`은 처음 N장으로 설정만 확인합니다.
`evaluate.py`는 `--force` 없이는 이전 결과를 덮어쓰지 않습니다.
`inference.py`는 같은 이름의 기존 출력 파일을 덮어쓰며 `--force` 옵션을 지원하지 않습니다.

## 이미지 한 장 추론

```bash
python inference.py --source /path/to/image.jpg --framework mxq
python inference.py --source /path/to/image.jpg --framework onnx
```

## 동기·비동기 연속 추론

```bash
python inference_sync.py --source /path/to/images --core-mode multi
python inference_async.py --source /path/to/images --core-mode multi --queue-depth 4
```

두 버전은 모델을 한 번 로드하고 제한된 수의 이미지를 미리 전처리합니다. 비동기 버전은
qbruntime의 `infer_async`와 완료 future를 사용하며 입력 순서를 유지합니다.
`--preprocess-workers`로 전처리 병렬 수를 조절하고, `--save-outputs`로 이미지별 원시 출력과
해석 결과를 저장합니다. 기본값은 파일 저장 없이 처리량을 측정합니다. 처리량에는 전처리와
추론(출력 저장을 선택하면 후처리·저장도)이 포함되며 모델 로드는 제외됩니다.
`--limit`과 `--repeat`로 같은 입력에서 비교하세요. 비동기 버전이 항상 빠르지는 않습니다.

모든 script는 `--help`를 지원합니다.
