from determined import DistributedContext


class ModelParallelUnit:
    """
    This class contains the functions we expect in order to
    accurately carry out parallel training.  The base class
    returns information for just data parallel training.
    You need to subclass and override the functions for custom
    model parallel training and pass it to the pytorch context
    by calling context.wrap_mpu(mpu).

    The class is modeled after the methods expected from
    a model parallel unit by deepspeed.
    """

    def __init__(self, dist_context: DistributedContext):
        self.dist_context = dist_context

    def get_global_rank(self) -> int:
        return self.dist_context.get_rank()

    def get_data_parallel_rank(self) -> int:
        # Which pipeline this rank resides in.
        return self.dist_context.get_rank()

    def get_data_parallel_world_size(self) -> int:
        # The number of pipelines.
        return self.dist_context.get_size()

    def should_report_metrics(self) -> bool:
        # Whether the rank should return metrics from training and evaluation steps.
        # This is usually true for ranks in the last stage of pipeline parallel or
        # model parallel training.
        return True

    def should_build_data_loader(self) -> bool:
        # Whether the rank should build a data loader.
        # This is usually true for ranks in the first or last stage of
        # pipeline parallel or model parallel training.
        return True
