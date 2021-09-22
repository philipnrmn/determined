from typing import Any, cast

import torch
from deepspeed.runtime.pipe.topology import PipelineParallelGrid

from determined.pytorch import ModelParallelUnit


class BatchDataset(torch.utils.data.Dataset):
    def __init__(self, batch: Any) -> None:
        self.batch = batch

    def __getitem__(self, idx: int) -> Any:
        if isinstance(self.batch, dict):
            return {k: self.batch[k][idx] for k in self.batch.keys()}
        elif isinstance(self.batch, list):
            return [entry[idx] for entry in self.batch]
        else:
            return self.batch[idx]

    def __len__(self) -> int:
        if isinstance(self.batch, dict):
            keys = list(self.batch.keys())
            return len(self.batch[keys[0]])
        if isinstance(self.batch, list):
            return len(self.batch[0])
        return len(self.batch)


class DeepSpeedMPU(ModelParallelUnit):
    def __init__(self, mpu: PipelineParallelGrid) -> None:
        self.mpu = mpu

    def get_global_rank(self) -> int:
        return cast(int, self.mpu.get_global_rank())

    def get_data_parallel_rank(self) -> int:
        return cast(int, self.mpu.get_data_parallel_rank())

    def get_data_parallel_world_size(self) -> int:
        return cast(int, self.mpu.get_data_parallel_world_size())

    def is_first_pipeline_stage(self) -> bool:
        return cast(int, self.mpu.get_pipe_parallel_rank()) == 0

    def is_last_pipeline_stage(self) -> bool:
        return cast(int, self.mpu.get_pipe_parallel_rank()) == (
            cast(int, self.mpu.get_pipe_parallel_world_size()) - 1
        )

    def should_report_metrics(self) -> bool:
        return self.is_last_pipeline_stage()

    def should_build_data_loader(self) -> bool:
        return cast(int, self.mpu.get_slice_parallel_rank()) == 0 and (
            self.is_first_pipeline_stage() or self.is_last_pipeline_stage()
        )
