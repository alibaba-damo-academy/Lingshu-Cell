"""Workflow utility CLI scripts.

Modules:
    wait_checkpoint: Poll for a checkpoint (EMA or main) to appear on disk.
        Usage: python -m workflows.utils.wait_checkpoint --training-dir ... --iteration ...
               python -m workflows.utils.wait_checkpoint --training-dir ... --iteration ... --type ckpt
    merge_with_control: Merge sample subtask outputs with control logic.
"""
