from qbcompiler import (
    BitConfig,
    CalibrationConfig,
    EquivalentTransformationConfig,
    LlmConfig,
    OptqConfig,
    ResourceManagementConfig,
    SearchWeightScaleConfig,
)


def get_configs() -> dict:
    return {
        "calibration_config": CalibrationConfig(output=0, mode=0),

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
            qk=EquivalentTransformationConfig.Qk(apply=False),
            ud=EquivalentTransformationConfig.Ud(apply=True, smoothing_factor=0.8),
            vo=EquivalentTransformationConfig.Vo(apply=True),
            spin_r1=EquivalentTransformationConfig.SpinR1(apply=True),
            spin_r2=EquivalentTransformationConfig.SpinR2(apply=True),
            optimize_ffn=EquivalentTransformationConfig.OptimizeFfn(apply=True),
        ),

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

        "optq_config": OptqConfig(
            apply=True,
            attributes=OptqConfig.Attributes(
                act_order=True,
                block_size=128,
                perc_damp=0.01,
            ),
        ),
    }
