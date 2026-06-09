"""Run nnU-Net training directly via Python API"""
import os, sys, time
import torch

# PyTorch 2.6+ 默认 weights_only=True，与 nnU-Net checkpoint 不兼容
# Monkey-patch: 恢复旧版 torch.load 默认行为 (weights_only=False)
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

def main():
    os.environ["nnUNet_raw"] = "D:\\Science\\Medical_AI_Robustness\\nnUNet_data\\raw"
    os.environ["nnUNet_preprocessed"] = "D:\\Science\\Medical_AI_Robustness\\nnUNet_data\\preprocessed"
    os.environ["nnUNet_results"] = "D:\\Science\\Medical_AI_Robustness\\nnUNet_data\\results"
    os.environ["nnUNet_n_proc_DA"] = "0"

    from nnunetv2.run.run_training import run_training

    print("Starting nnU-Net training via Python API...")
    print(f"PID: {os.getpid()}")
    sys.stdout.flush()

    # Same args as CLI: dataset_id=800, config=2d, fold=0
    run_training(
        dataset_name_or_id="800",
        configuration="2d",
        fold=0,
        trainer_class_name="nnUNetTrainer",
        plans_identifier="nnUNetPlans",
        use_compressed_data=False,
        disable_checkpointing=False,
        continue_training=True,  # 从 checkpoint 续训
        only_run_validation=False,
        pretrained_weights=None,
        export_validation_probabilities=False,
        val_with_best=False,
        num_gpus=1,
    )

    print("Training completed!")

if __name__ == "__main__":
    main()
