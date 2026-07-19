from qbcompiler import (
    BitConfig,
    CalibrationConfig,
    EquivalentTransformationConfig,
    LlmConfig,
    ResourceManagementConfig,
)


def get_configs() -> dict:
    return {
        "calibration_config": CalibrationConfig(output=0, mode=0),

        "bit_config": BitConfig(
            layer_overrides=BitConfig.LayerOverrides(
                activation_16bits=[
                    "inputs_embeds/reshape",
                    "deepstack_visual_embeds_0",
                ],
            ),
        ),

        "resource_management_config": ResourceManagementConfig(
            weight_dtype="float32",
            use_gpu_only_for_calibration=True,
            weight_memory=ResourceManagementConfig.WeightMemory(method=1),
        ),

        "llm_config": LlmConfig(
            apply=True,
            attributes=LlmConfig.Attributes(
                calibration=LlmConfig.Attributes.Calibration(use_full_seq_length=True),
            ),
        ),

        "equivalent_transformation_config": EquivalentTransformationConfig(
            qk=EquivalentTransformationConfig.Qk(apply=True),
            qk_rotation=EquivalentTransformationConfig.QkRotation(apply=True),
            ud=EquivalentTransformationConfig.Ud(apply=True),
            norm_conv=EquivalentTransformationConfig.NormConv(apply=True),
            vo=EquivalentTransformationConfig.Vo(apply=True),
            spin_r1=EquivalentTransformationConfig.SpinR1(apply=True),
            spin_r2=EquivalentTransformationConfig.SpinR2(apply=True),
            flatten_quant=EquivalentTransformationConfig.FlattenQuant(apply=True),
            optimize_ffn=EquivalentTransformationConfig.OptimizeFfn(apply=True),
        ),

        "optq_config": None,
    }
