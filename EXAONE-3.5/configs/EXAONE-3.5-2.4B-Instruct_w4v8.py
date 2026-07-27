from qbcompiler import (
    BitConfig,
    CalibrationConfig,
    EquivalentTransformationConfig,
    LlmConfig,
    ResourceManagementConfig,
    SearchWeightScaleConfig,
)


def get_configs() -> dict:
    """Return mxq_compile config kwargs for EXAONE-3.5-2.4B-Instruct / w4v8."""
    return {
        "resource_management_config": ResourceManagementConfig(
            weight_dtype="float16",
            weight_memory=ResourceManagementConfig.WeightMemory(method=0),
        ),
        "calibration_config": CalibrationConfig(mode=0),
        # weight bits from the "w4v8" weight_type.
        "bit_config": BitConfig(
            transformer=BitConfig.Transformer(
                weight=BitConfig.Transformer.Weight(
                    query=4,
                    key=4,
                    value=8,
                    output=4,
                    ffn=4,
                    head=4,
                ),
            ),
        ),
        # SpinQuant rotations / equivalent transforms per the "R1R2QKUD" preset.
        "equivalent_transformation_config": EquivalentTransformationConfig(
            seed=0,
            apply_hadamard_rotation_matrix=True,
            spin_r1=EquivalentTransformationConfig.SpinR1(apply=True),
            spin_r2=EquivalentTransformationConfig.SpinR2(apply=True),
            qk=EquivalentTransformationConfig.Qk(apply=True),
            qk_rotation=EquivalentTransformationConfig.QkRotation(apply=True),
            ud=EquivalentTransformationConfig.Ud(apply=True, learn=True),
            norm_conv=EquivalentTransformationConfig.NormConv(apply=False),
            flatten_quant=EquivalentTransformationConfig.FlattenQuant(apply=False),
            optimize_ffn=EquivalentTransformationConfig.OptimizeFfn(apply=True),
        ),
        # Weight-scale search: on for w4v8, off for w8 (sws_apply from weight_type).
        "search_weight_scale_config": SearchWeightScaleConfig(
            apply=True,
            transformer=SearchWeightScaleConfig.Transformer(
                query=True,
                key=True,
                value=True,
                out=True,
                ffn=True,
            ),
        ),
        "llm_config": LlmConfig(
            apply=True,
            attributes=LlmConfig.Attributes(
                max_data_length=256,
                max_sequence_length=4096,
                max_cache_length=4096,
                max_core_data_length=256,
                calibration=LlmConfig.Attributes.Calibration(use_full_seq_length=True),
                runtime=LlmConfig.Attributes.Runtime(batch_size=1),
            ),
        ),
    }
