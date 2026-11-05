# SDK 1.4 compatibility

## Reproduce with the recorded SDK

The 102 Models below have published `compile_config.json` files that **fail
configuration loading with both qbcompiler 1.4.0 and 1.4.1**. The recorded compiler
version is listed per Model; all 102 were recorded with qbcompiler 1.3.0. To
reproduce a recorded result, use that Model's recorded SDK and Toolchain together
with its published reproduction instructions, rather than substituting 1.4.0 or
1.4.1. The SDK is not included in this repository.

This notice does not change any artifact, per-Model configuration, recorded result,
or approval. It is not a migration recipe for the rejected fields. Do not simply
delete fields or change SDK versions and assume the recorded result is preserved.
A newer SDK requires a separately validated configuration, recompilation, and
accuracy evaluation.

## Measured loader compatibility

All 337 published `compile_config.json` files at
[arxiv commit `9eefd2860eaa97a279e372e53dbbd02c50089ff8`](https://github.com/mobilint/mblt-model-arxiv/tree/9eefd2860eaa97a279e372e53dbbd02c50089ff8)
were checked without modifying their bytes. The table below was generated from
that audit's per-Model CSV, not entered manually. Both releases produced the same
results: **235 accepted, 102 rejected**. For accepted configurations, the typed
bias correction setting was `true` in 222 and `false` in 13; these counts do not
establish that any Model compiles or reproduces its recorded accuracy. A rejected
configuration has no effective bias correction setting: its value is **N/A**.

The CPU-only check executed the unmodified `qbcompiler/configs/models.py` source
from each released wheel and called its original `CompileConfig.from_file` on
each published file, including its grouped-JSON flattening before typed validation.
It did not execute the full compiler package, compile a Model, run inference, or
measure accuracy. The Python/Pydantic/YAML versions were 3.10.12/2.12.5/6.0.3.
This is a configuration-loader check, **not compilation, accuracy, or reproduction
evidence**. Successful loading alone does not establish SDK compatibility for a
complete reproduction.

Wheel SHA256 values:

- qbcompiler 1.4.0: `c361bbb1d5be21f81f8eb4dd413aa12516c9d4f78db6a06d99543fa54b33c994`
- qbcompiler 1.4.1: `2f96aca017c67571dfa16c5b582835e335df93213c49758beb81ee4ef9529ffb`

The audit proof SHA256 is
`92964c927d688e311450bf498a732a9f85f5a4022d61961f8c3ef40ad331d4d5`;
the source CSV SHA256 is
`c7a99d262c69a98d21350700a86f31a4f1ced25e5d5135e60c6cee71e277deb8`.

### Rejected fields

Each listed field was rejected with typed validation error `extra_forbidden`.
The paths below are the loader's error locations; grouped JSON source paths are
shown where they differ:

- `resourceManagement.useGPUOnlyForCalibration`: same source path.
- `calibration.optimizeLut`: source `quantization.calibration.optimizeLut`.
- `hessianQuant.accumulationDevice`: source `advancedQuantization.hessianQuant.accumulationDevice`.
- `inputShapeDict`: same source path.

20 Models fail on three fields, 52 on all four, and 30 on
`calibration.optimizeLut` plus `inputShapeDict`. These 102 include 64 legacy BC-OFF,
8 legacy BC-ON, and 30 modern-BC configurations. The table lists the exact rejected
fields for each Model. Every row is rejected by **both 1.4.0 and 1.4.1**; effective
BC is **N/A** in every row. The final column identifies only the 64 OFF configurations
covered by the additional BC guidance below, not every configuration with BC off.

## Additional bias correction guidance for the 64 marked Models

The 64 rows marked `yes` retain the legacy
`advancedQuantization.layerBiasCorrection` block with `apply: false`.
The original `CompileConfig.from_file` loader in 1.4.0 and 1.4.1 discards that legacy
key while flattening grouped JSON; it is not one of the rejected fields above.
The modern `biasCorrection.apply` setting defaults to `true`. A minimal otherwise
valid configuration containing only the legacy OFF block can therefore load with
BC enabled. **The actual published configurations listed here do not load:** their
other rejected fields remain. Replacing the BC block alone does not make them load
and does not establish equivalent accuracy.

For a separately validated migration of a marked OFF configuration, replace the
entire `advancedQuantization.layerBiasCorrection` block with
`advancedQuantization.biasCorrection: {"apply": false}` before loading. In JSON,
the replacement within `advancedQuantization` is

```json
"biasCorrection": {"apply": false}
```

Do not keep both blocks or carry legacy attributes into the replacement block.
This preserves only the intended OFF setting; it does not resolve the other
rejected fields. This OFF instruction is not a migration instruction for
configurations with bias correction enabled or for unmarked rows. Recompile and
evaluate with the intended SDK before relying on its result.

Newer releases will set bias correction explicitly; a Model is updated only when
its new release reproduces with equal or better accuracy. Existing per-Model files
and recorded results remain unchanged by this notice.

## Affected Models

| Model | Recorded qbcompiler | Rejected fields in both 1.4.0 and 1.4.1 | Effective BC | Legacy BC-OFF guidance |
| --- | --- | --- | --- | --- |
| `ArcFace_IResNet_100` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice` | N/A (rejected) | yes |
| `ArcFace_IResNet_18` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice` | N/A (rejected) | yes |
| `ArcFace_IResNet_34` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice` | N/A (rejected) | yes |
| `ArcFace_IResNet_50` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice` | N/A (rejected) | yes |
| `CAFormer_B36` | `1.3.0` | `calibration.optimizeLut`, `inputShapeDict` | N/A (rejected) | no |
| `CAFormer_S18` | `1.3.0` | `calibration.optimizeLut`, `inputShapeDict` | N/A (rejected) | no |
| `CLIP_ReID_RN50_Baseline_DukeMTMC` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice` | N/A (rejected) | yes |
| `CLIP_ReID_RN50_Baseline_Market1501` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice` | N/A (rejected) | yes |
| `CLIP_ReID_RN50_Baseline_OccDuke` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice` | N/A (rejected) | yes |
| `CLIP_ReID_RN50_DukeMTMC` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice` | N/A (rejected) | yes |
| `CLIP_ReID_RN50_Market1501` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice` | N/A (rejected) | yes |
| `CLIP_ReID_RN50_OccDuke` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice` | N/A (rejected) | yes |
| `CLIP_ReID_ViT_B_16_Baseline_DukeMTMC` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice` | N/A (rejected) | yes |
| `CLIP_ReID_ViT_B_16_Baseline_Market1501` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice` | N/A (rejected) | yes |
| `CLIP_ReID_ViT_B_16_Baseline_OccDuke` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice` | N/A (rejected) | yes |
| `CLIP_ReID_ViT_B_16_DukeMTMC` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice` | N/A (rejected) | yes |
| `CLIP_ReID_ViT_B_16_Market1501` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice` | N/A (rejected) | yes |
| `CLIP_ReID_ViT_B_16_OccDuke` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice` | N/A (rejected) | yes |
| `CoAtNet_1_RW_224` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `ConvFormer_S18` | `1.3.0` | `calibration.optimizeLut`, `inputShapeDict` | N/A (rejected) | no |
| `ConvFormer_S36` | `1.3.0` | `calibration.optimizeLut`, `inputShapeDict` | N/A (rejected) | no |
| `ConvNeXt_Large_IMAGENET1K_V1` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | no |
| `CosFace_IResNet_100` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice` | N/A (rejected) | yes |
| `CosFace_IResNet_18` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice` | N/A (rejected) | yes |
| `CosFace_IResNet_34` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice` | N/A (rejected) | yes |
| `CosFace_IResNet_50` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice` | N/A (rejected) | yes |
| `DeiT3_Base_Patch16_384` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `DeiT3_Large_Patch16_224` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | no |
| `DeiT3_Small_Patch16_224` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `DeiT3_Small_Patch16_384` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `DeiT_Base_Patch16_384` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `DenseNet169_IMAGENET1K_V1` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `DenseNet201_IMAGENET1K_V1` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `EfficientNet_B3_IMAGENET1K_V1` | `1.3.0` | `calibration.optimizeLut`, `inputShapeDict` | N/A (rejected) | no |
| `EfficientNet_V2_L_IMAGENET1K_V1` | `1.3.0` | `calibration.optimizeLut`, `inputShapeDict` | N/A (rejected) | no |
| `FlexiVit_Base` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `FlexiVit_Large` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | no |
| `GELANe` | `1.3.0` | `calibration.optimizeLut`, `inputShapeDict` | N/A (rejected) | no |
| `GoogLeNet_IMAGENET1K_V1` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `RegNet_Y_32GF_IMAGENET1K_V2` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | no |
| `RegNet_Y_400MF_IMAGENET1K_V2` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `RegNet_Y_800MF_IMAGENET1K_V2` | `1.3.0` | `calibration.optimizeLut`, `inputShapeDict` | N/A (rejected) | no |
| `RepViT_M2` | `1.3.0` | `calibration.optimizeLut`, `inputShapeDict` | N/A (rejected) | no |
| `RepViT_M2_3` | `1.3.0` | `calibration.optimizeLut`, `inputShapeDict` | N/A (rejected) | no |
| `ResNeXt50_32X4D_IMAGENET1K_V2` | `1.3.0` | `calibration.optimizeLut`, `inputShapeDict` | N/A (rejected) | no |
| `ResNet18_IMAGENET1K_V1` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | no |
| `ResNet50_IMAGENET1K_V2` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `VGG16_IMAGENET1K_V1` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `ViT_Base_Patch16_224` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `ViT_Base_Patch16_RPN_224` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `ViT_L_16_IMAGENET1K_SWAG_E2E_V1` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `ViT_L_16_IMAGENET1K_SWAG_LINEAR_V1` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `ViT_Large_Patch16_224` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `ViT_Large_Patch32_384` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `ViT_Medium_Patch16_GAP_256` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | no |
| `ViT_RelPos_Medium_Patch16_224` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `ViT_RelPos_Medium_Patch16_RPN_224` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `ViT_RelPos_Small_Patch16_224` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `ViT_SRelPos_Medium_Patch16_224` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `ViT_SRelPos_Small_Patch16_224` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `ViT_Small_Patch16_224` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `ViT_Small_Patch32_224` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `YOLO11l_obb` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `YOLO11l_seg` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `YOLO11m_obb` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `YOLO11m_seg` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `YOLO11n_obb` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `YOLO11s_obb` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `YOLO11x_cls` | `1.3.0` | `calibration.optimizeLut`, `inputShapeDict` | N/A (rejected) | no |
| `YOLO11x_obb` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `YOLO11x_pose` | `1.3.0` | `calibration.optimizeLut`, `inputShapeDict` | N/A (rejected) | no |
| `YOLO12l_face` | `1.3.0` | `calibration.optimizeLut`, `inputShapeDict` | N/A (rejected) | no |
| `YOLO12m_face` | `1.3.0` | `calibration.optimizeLut`, `inputShapeDict` | N/A (rejected) | no |
| `YOLO12s_seg` | `1.3.0` | `calibration.optimizeLut`, `inputShapeDict` | N/A (rejected) | no |
| `YOLO12x` | `1.3.0` | `calibration.optimizeLut`, `inputShapeDict` | N/A (rejected) | no |
| `YOLO12x_seg` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `YOLO26l` | `1.3.0` | `calibration.optimizeLut`, `inputShapeDict` | N/A (rejected) | no |
| `YOLO26n_obb` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `YOLO26s_obb` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `YOLO26x_depth` | `1.3.0` | `calibration.optimizeLut`, `inputShapeDict` | N/A (rejected) | no |
| `YOLOv10x` | `1.3.0` | `calibration.optimizeLut`, `inputShapeDict` | N/A (rejected) | no |
| `YOLOv3_spp` | `1.3.0` | `calibration.optimizeLut`, `inputShapeDict` | N/A (rejected) | no |
| `YOLOv5l6u` | `1.3.0` | `calibration.optimizeLut`, `inputShapeDict` | N/A (rejected) | no |
| `YOLOv5m_cls` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `YOLOv5m_seg` | `1.3.0` | `calibration.optimizeLut`, `inputShapeDict` | N/A (rejected) | no |
| `YOLOv5n_0_5_face` | `1.3.0` | `calibration.optimizeLut`, `inputShapeDict` | N/A (rejected) | no |
| `YOLOv5x6` | `1.3.0` | `calibration.optimizeLut`, `inputShapeDict` | N/A (rejected) | no |
| `YOLOv7_face` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `YOLOv7w6` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `YOLOv7x` | `1.3.0` | `calibration.optimizeLut`, `inputShapeDict` | N/A (rejected) | no |
| `YOLOv8l_obb` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | no |
| `YOLOv8m_face` | `1.3.0` | `calibration.optimizeLut`, `inputShapeDict` | N/A (rejected) | no |
| `YOLOv8m_obb` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `YOLOv8m_seg` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `YOLOv8n_obb` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `YOLOv8n_pose` | `1.3.0` | `calibration.optimizeLut`, `inputShapeDict` | N/A (rejected) | no |
| `YOLOv8s_obb` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `YOLOv8s_seg` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `YOLOv8x_obb` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | no |
| `YOLOv8x_pose` | `1.3.0` | `calibration.optimizeLut`, `inputShapeDict` | N/A (rejected) | no |
| `YOLOv8x_seg` | `1.3.0` | `resourceManagement.useGPUOnlyForCalibration`, `calibration.optimizeLut`, `hessianQuant.accumulationDevice`, `inputShapeDict` | N/A (rejected) | yes |
| `YOLOv9c` | `1.3.0` | `calibration.optimizeLut`, `inputShapeDict` | N/A (rejected) | no |
