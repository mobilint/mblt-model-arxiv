from qbcompiler import (
    BitConfig,
    CalibrationConfig,
    EquivalentTransformationConfig,
    LlmConfig,
    ResourceManagementConfig,
)


def get_configs(rotation_matrix_path) -> dict:
    # rotation_matrix_path: the decoder's SpinR1 matrix, consumed via head_out_ch_rotation.
    # Supplied by mxq_compile_encoder.py (derived from --model, or --rotation-matrix-path),
    # so it is not hardcoded here. Run the decoder first to produce it.
    return {
        "calibration_config": CalibrationConfig(output=0, mode=1),

        "bit_config": BitConfig(
            layer_overrides=BitConfig.LayerOverrides(
                activation_16bits=[
                    "model_merger_linear_fc2",
                    "model_deepstack_merger_list_0_linear_fc2",
                    "model_deepstack_merger_list_1_linear_fc2",
                    "model_deepstack_merger_list_2_linear_fc2",
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
            ud=EquivalentTransformationConfig.Ud(apply=True),
            vo=EquivalentTransformationConfig.Vo(apply=True),
            head_out_ch_rotation=EquivalentTransformationConfig.HeadOutChRotation(
                apply=True,
                matrix_path=rotation_matrix_path,
            ),
            spin_r1=EquivalentTransformationConfig.SpinR1(apply=False),
            spin_r2=EquivalentTransformationConfig.SpinR2(apply=True),
            optimize_ffn=EquivalentTransformationConfig.OptimizeFfn(apply=True),
        ),
    }
