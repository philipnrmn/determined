import logging
import pathlib
import pickle
import random
import time
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch

import determined as det
from determined import horovod, layers, pytorch, util, workload
from determined.common import check, experimental, storage
from determined.common.api import certs
from determined.common.api.analytics import send_analytics
from determined.horovod import hvd
from determined.util import has_param

# Apex is included only for GPU trials.
try:
    import apex
except ImportError:
    pass


class PyTorchTrialController(det.TrialController):
    def __init__(self, trial_inst: det.Trial, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        send_analytics("PyTorchTrial Created")

        check.is_instance(trial_inst, PyTorchTrial, "PyTorchTrialController needs an PyTorchTrial")
        self.trial = cast(PyTorchTrial, trial_inst)
        self.context = cast(pytorch.PyTorchTrialContext, self.context)
        self.context._set_determined_profiler(self.prof)
        if torch.cuda.is_available():
            self.prof._set_sync_device(self._sync_device)
        self.callbacks = self.trial.build_callbacks()

        check.gt_eq(
            len(self.context.models),
            1,
            "Must have at least one model. "
            "This might be caused by not wrapping your model with wrap_model()",
        )
        check.gt_eq(
            len(self.context.optimizers),
            1,
            "Must have at least one optimizer. "
            "This might be caused by not wrapping your optimizer with wrap_optimizer()",
        )
        self._check_evaluate_implementation()

        # Set model parallel configuration before building dataloaders.
        self.mpu = (
            pytorch.ModelParallelUnit(self.context.distributed)
            if self.context._mpu is None
            else self.context._mpu
        )

        # Validation loader will be undefined on process ranks > 0
        # when the user defines `validate_full_dataset()`.
        self.validation_loader = None  # type: Optional[torch.utils.data.DataLoader]
        self.num_validation_batches = None  # type: Optional[int]
        self._set_data_loaders()

        self.wlsq = None  # type: Optional[layers.WorkloadSequencer]
        if self.workloads is None:
            session = experimental.Session(None, None, None, certs.cli_cert)
            self.workloads, self.wlsq = layers.make_compatibility_workloads(
                session, self.env, self.context.distributed
            )

    @staticmethod
    def pre_execute_hook(env: det.EnvContext, hvd_config: horovod.HorovodContext) -> None:
        # Initialize the correct horovod.
        if env.hparams.get("deepspeed", False):
            hvd_config.use = False
        if hvd_config.use:
            hvd.require_horovod_type("torch", "PyTorchTrial is in use.")
            hvd.init()

        PyTorchTrialController._set_random_seeds(env.trial_seed)

    @staticmethod
    def _set_random_seeds(seed: int) -> None:
        # Set identical random seeds on all training processes.
        # When using horovod, each worker will start at a unique
        # offset in the dataset, ensuring it's processing a unique
        # training batch.
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        # TODO(Aaron): Add flag to enable determinism.
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    @staticmethod
    def from_trial(*args: Any, **kwargs: Any) -> det.TrialController:
        return PyTorchTrialController(*args, **kwargs)

    @staticmethod
    def supports_mixed_precision() -> bool:
        return True

    @staticmethod
    def supports_averaging_training_metrics() -> bool:
        return True

    def _check_evaluate_implementation(self) -> None:
        """
        Check if the user has implemented evaluate_batch
        or evaluate_full_dataset.
        """
        logging.debug(f"Evaluate_batch_defined: {self._evaluate_batch_defined()}.")
        logging.debug(f"Evaluate full dataset defined: {self._evaluate_full_dataset_defined()}.")
        check.not_eq(
            self._evaluate_batch_defined(),
            self._evaluate_full_dataset_defined(),
            "Please define exactly one of: `evaluate_batch()` or `evaluate_full_dataset()`. "
            "For most use cases `evaluate_batch()` is recommended because "
            "it can be parallelized across all devices.",
        )

    def _evaluate_batch_defined(self) -> bool:
        return util.is_overridden(self.trial.evaluate_batch, PyTorchTrial)

    def _evaluate_full_dataset_defined(self) -> bool:
        return util.is_overridden(self.trial.evaluate_full_dataset, PyTorchTrial)

    def _set_data_loaders(self) -> None:
        skip_batches = self.env.latest_batch
        nreplicas = self.mpu.get_data_parallel_world_size()
        rank = self.mpu.get_data_parallel_rank()

        def _dataset_repro_warning(fn: str, data_obj: Any) -> str:
            return (
                f"{fn}() returned an instance of {type(data_obj).__name__}, which is not a "
                "subclass of det.pytorch.DataLoader.  For most non-Iterable DataSets, "
                "det.pytorch.DataLoader is a drop-in replacement for torch.utils.data.DataLoader "
                "but which offers easy and transparent reproducibility in Determined experiments. "
                "It is highly recommended that you use det.pytorch.DataLoader if possible.  If "
                "not, you can disable this check by calling "
                "context.experimental.disable_dataset_reproducibility_checks() at some point in "
                "your trial's __init__() method."
            )

        train_data = self.trial.build_training_data_loader()
        if isinstance(train_data, pytorch.DataLoader):
            self.training_loader = train_data.get_data_loader(
                repeat=True, skip=skip_batches, num_replicas=nreplicas, rank=rank
            )
        else:
            if train_data is not None:
                # Non-determined DataLoader; ensure the user meant to do this.
                if not self.context.experimental._data_repro_checks_disabled:
                    raise RuntimeError(
                        _dataset_repro_warning("build_training_data_loader", train_data)
                    )
            self.training_loader = train_data

        self.context._epoch_len = len(self.training_loader) if train_data is not None else None
        all_epoch_lens = self.context.distributed._zmq_gather(self.context._epoch_len)
        if self.is_chief:
            all_epoch_lens = [le for le in all_epoch_lens if le is not None]
            if min(all_epoch_lens) < max(all_epoch_lens):
                logging.warning("Training dataloader length inconsistent across ranks.")
            self.context._epoch_len = min(all_epoch_lens)
        self.context._epoch_len = self.context.distributed._zmq_broadcast(self.context._epoch_len)

        validation_data = self.trial.build_validation_data_loader()
        if self._evaluate_batch_defined():
            if isinstance(validation_data, pytorch.DataLoader):
                self.validation_loader = validation_data.get_data_loader(
                    repeat=False, skip=0, num_replicas=nreplicas, rank=rank
                )
            else:
                if validation_data is not None:
                    # Non-determined DataLoader; ensure the user meant to do this.
                    if not self.context.experimental._data_repro_checks_disabled:
                        raise RuntimeError(
                            _dataset_repro_warning("build_validation_data_loader", validation_data)
                        )
                self.validation_loader = validation_data
        elif self.is_chief:
            if isinstance(validation_data, pytorch.DataLoader):
                self.validation_loader = validation_data.get_data_loader(
                    repeat=False, skip=0, num_replicas=1, rank=0
                )
            else:
                # Non-determined DataLoader; ensure the user meant to do this.
                if not self.context.experimental._data_repro_checks_disabled:
                    raise RuntimeError(
                        _dataset_repro_warning("build_validation_data_loader", validation_data)
                    )
                self.validation_loader = validation_data

    def run(self) -> None:
        # We create the training_iterator here rather than in __init__ because we have to be careful
        # to trigger its shutdown explicitly, to avoid hangs in when the user is using
        # multiprocessing-based parallelism for their dataloader.
        #
        # We create it before loading state because we don't want the training_iterator shuffling
        # values after we load state.
        self.training_iterator = (
            iter(self.training_loader) if self.training_loader is not None else None
        )

        try:
            # If a load path is provided load weights and restore the data location.
            if self.env.latest_checkpoint is not None:
                with self._generic._download_initial_checkpoint(
                    self.env.latest_checkpoint
                ) as load_path:
                    self._load(pathlib.Path(load_path))

            if self.hvd_config.use:
                hvd.broadcast_parameters(self.context._main_model.state_dict(), root_rank=0)
                for optimizer in self.context.optimizers:
                    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

            with self.prof:
                for callback in self.callbacks.values():
                    with self.prof.record_timing(
                        f"callbacks.{callback.__class__.__name__}.on_training_start"
                    ):
                        callback.on_training_start()
                self._run()

                for callback in self.callbacks.values():
                    with self.prof.record_timing(
                            f"callbacks.{callback.__class__.__name__}.trial_cleanup"
                    ):
                        callback.trial_cleanup()
        finally:
            # Explicitly trigger the training iterator's shutdown (which happens in __del__).
            # See the rather long note in pytorch/torch/utils/data/dataloader.py.
            if self.training_iterator is not None:
                del self.training_iterator

    def _run(self) -> None:
        assert self.workloads is not None
        for w, response_func in self.workloads:
            try:
                if w.kind == workload.Workload.Kind.RUN_STEP:
                    action = "training"
                    response = {
                        "metrics": self._train_for_step(
                            w.step_id,
                            w.num_batches,
                            w.total_batches_processed,
                        ),
                        "stop_requested": self.context.get_stop_requested(),
                    }  # type: workload.Response

                elif w.kind == workload.Workload.Kind.COMPUTE_VALIDATION_METRICS:
                    action = "validation"
                    response = {
                        "metrics": self._compute_validation_metrics(),
                        "stop_requested": self.context.get_stop_requested(),
                    }

                elif w.kind == workload.Workload.Kind.CHECKPOINT_MODEL:
                    action = "checkpointing"
                    if self.is_chief:
                        with self._generic._storage_mgr.store_path() as (storage_id, path):
                            # Broadcast checkpoint path to all ranks.
                            self.context.distributed._zmq_broadcast(path)
                            self._save(pathlib.Path(path))
                            # Wait for save to finish on all ranks and gather all resources to
                            # report to master.
                            all_resources = self.context.distributed._zmq_gather(
                                storage.StorageManager._list_directory(path)
                            )
                            resources = {k: v for d in all_resources for k, v in d.items()}
                            response = {
                                "uuid": storage_id,
                                "resources": resources,
                                "framework": f"torch-{torch.__version__}",
                                "format": "pickle",
                            }
                    else:
                        path = self.context.distributed._zmq_broadcast(None)
                        self._save(pathlib.Path(path))
                        _ = self.context.distributed._zmq_gather(
                            storage.StorageManager._list_directory(path)
                        )
                        response = {}

                else:
                    raise AssertionError("Unexpected workload: {}".format(w.kind))

            except det.InvalidHP as e:
                logging.info(f"Invalid hyperparameter exception during {action}: {e}")
                response = workload.InvalidHP()
            response_func(response)

    def get_epoch_idx(self, batch_id: int) -> int:
        self.context._epoch_len = cast(int, self.context._epoch_len)
        return batch_id // self.context._epoch_len

    def _average_training_metrics(
        self, per_batch_metrics: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Average training metrics across GPUs"""
        check.true(
            self.context.distributed.get_size() > 0,
            "Can only average training metrics in multi-GPU training.",
        )
        metrics_timeseries = util._list_to_dict(per_batch_metrics)

        # combined_timeseries is: dict[metric_name] -> 2d-array.
        # A measurement is accessed via combined_timeseries[metric_name][process_idx][batch_idx].
        combined_timeseries, combined_num_batches = self._combine_metrics_across_processes(
            metrics_timeseries, num_batches=len(per_batch_metrics)
        )

        if self.is_chief:
            # We can safely cast variables here because this is all happening on the chief, which
            # is where we gather metrics.
            combined_timeseries = cast(Dict[str, Any], combined_timeseries)
            combined_num_batches = cast(List[int], combined_num_batches)
            # If the value for a metric is a single-element array, the averaging process will
            # change that into just the element. We record what metrics are single-element arrays
            # so we can wrap them in an array later (for perfect compatibility with non-averaging
            # codepath).
            array_metrics = []
            for metric_name in combined_timeseries.keys():
                process_batches = cast(List[List[Any]], combined_timeseries[metric_name])
                if isinstance(process_batches[0][0], np.ndarray):
                    array_metrics.append(metric_name)
            combined_timeseries_type = Dict[str, List[List[Any]]]
            combined_timeseries = cast(combined_timeseries_type, combined_timeseries)
            num_batches = combined_num_batches[0]
            num_processes = self.mpu.get_data_parallel_world_size()
            averaged_metrics_timeseries = {}  # type: Dict[str, List]

            for metric_name in combined_timeseries.keys():
                averaged_metrics_timeseries[metric_name] = []
                for batch_idx in range(num_batches):
                    batch = [
                        combined_timeseries[metric_name][process_idx][batch_idx]
                        for process_idx in range(num_processes)
                    ]

                    np_batch = np.array(batch)
                    batch_avg = np.mean(np_batch[np_batch != None])  # noqa: E711
                    if metric_name in array_metrics:
                        batch_avg = np.array(batch_avg)
                    averaged_metrics_timeseries[metric_name].append(batch_avg)
            per_batch_metrics = util._dict_to_list(averaged_metrics_timeseries)
        return per_batch_metrics

    def _auto_step_lr_scheduler_per_batch(
        self, batch_idx: int, lr_scheduler: pytorch.LRScheduler
    ) -> None:
        """
        This function aims at automatically step a LR scheduler. It should be called per batch.
        """

        # Never step lr when we do not step optimizer.
        if not self.context._should_communicate_and_update():
            return

        if lr_scheduler._step_mode == pytorch.LRScheduler.StepMode.STEP_EVERY_BATCH:
            start_idx = batch_idx - self.hvd_config.aggregation_frequency + 1
            for i in range(start_idx, batch_idx + 1):
                if (i + 1) % lr_scheduler._frequency == 0:
                    lr_scheduler.step()
        elif lr_scheduler._step_mode == pytorch.LRScheduler.StepMode.STEP_EVERY_OPTIMIZER_STEP:
            if (batch_idx + 1) % lr_scheduler._frequency == 0:
                lr_scheduler.step()
        elif lr_scheduler._step_mode == pytorch.LRScheduler.StepMode.STEP_EVERY_EPOCH:
            # We will step if the next optimizer step will land in the next epoch.
            epoch_idx = self.get_epoch_idx(batch_idx)
            next_steppable_batch = batch_idx + self.hvd_config.aggregation_frequency
            next_batch_epoch_idx = self.get_epoch_idx(next_steppable_batch)
            for e in range(epoch_idx, next_batch_epoch_idx):
                if (e + 1) % lr_scheduler._frequency == 0:
                    lr_scheduler.step()

    def _should_update_scaler(self) -> bool:
        if not self.context._scaler or not self.context.experimental._auto_amp:
            return False
        if self.hvd_config.use:
            return self.context._should_communicate_and_update()  # type: ignore
        return True

    def _train_for_step(
        self, step_id: int, num_batches: int, total_batches_processed: int
    ) -> workload.Response:
        self.prof.set_training(True)
        check.gt(step_id, 0)
        self.context.reset_reducers()

        # Set the behavior of certain layers (e.g., dropout) that are different
        # between training and inference.
        for model in self.context.models:
            model.train()

        start = total_batches_processed
        end = start + num_batches

        per_batch_metrics = []  # type: List[Dict]
        num_inputs = 0

        for batch_idx in range(start, end):
            batch_start_time = time.time()
            self.prof.update_batch_idx(batch_idx)
            if self.training_iterator is not None:
                with self.prof.record_timing("dataloader_next", requires_sync=False):
                    batch = next(self.training_iterator)
                batch_inputs = self.trial.get_batch_length(batch)

                if self.context.experimental._auto_to_device:
                    with self.prof.record_timing("to_device", accumulate=True):
                        batch = self.context.to_device(batch)
            else:
                batch = None
                batch_inputs = None
            if self.is_chief:
                assert batch_inputs > 0, "Batch must be non-empty on chief GPU."
            batch_inputs = self.context.distributed._zmq_broadcast(batch_inputs)

            self.context._current_batch_idx = batch_idx
            if self.context.is_epoch_start():
                for callback in self.callbacks.values():
                    with self.prof.record_timing(
                        f"callbacks.{callback.__class__.__name__}.on_training_epoch_start"
                    ):
                        callback.on_training_epoch_start()
            self.context._loss_ids = {}

            with self.prof.record_timing("train_batch", requires_sync=False):
                if self.context.profiler:
                    with self.context.profiler as torch_profiler:
                        tr_metrics = self.trial.train_batch(
                            batch=batch,
                            epoch_idx=self.get_epoch_idx(batch_idx),
                            batch_idx=batch_idx,
                        )
                        torch_profiler.step()
                else:
                    tr_metrics = self.trial.train_batch(
                        batch=batch,
                        epoch_idx=self.get_epoch_idx(batch_idx),
                        batch_idx=batch_idx,
                    )

            if self._should_update_scaler():
                self.context._scaler.update()

            # Step learning rate of a pytorch.LRScheduler.
            with self.prof.record_timing("step_lr_schedulers"):
                for lr_scheduler in self.context.lr_schedulers:
                    self._auto_step_lr_scheduler_per_batch(batch_idx, lr_scheduler)

            if self.mpu.should_report_metrics():
                if isinstance(tr_metrics, torch.Tensor):
                    tr_metrics = {"loss": tr_metrics}
                check.is_instance(
                    tr_metrics,
                    dict,
                    "train_batch() must return a dictionary "
                    f"mapping string names to Tensor metrics, got {type(tr_metrics)}",
                )

                with self.prof.record_timing("from_device"):
                    for name, metric in tr_metrics.items():
                        # Convert PyTorch metric values to NumPy, so that
                        # `det.util.encode_json` handles them properly without
                        # needing a dependency on PyTorch.
                        if isinstance(metric, torch.Tensor):
                            metric = metric.cpu().detach().numpy()
                        tr_metrics[name] = metric
                    per_batch_metrics.append(tr_metrics)

                num_inputs += batch_inputs
                batch_dur = time.time() - batch_start_time
                samples_per_second = (
                    self.mpu.get_data_parallel_world_size() * batch_inputs / batch_dur
                )
                self.prof.record_metric("samples_per_second", samples_per_second)

        # Aggregate and reduce training metrics from all the training processes.
        if self.context.experimental._use_deepspeed or (
            self.hvd_config.use and self.hvd_config.average_training_metrics
        ):
            with self.prof.record_timing("average_training_metrics"):
                per_batch_metrics = self._average_training_metrics(per_batch_metrics)
        if self.mpu.get_data_parallel_world_size() > 1:
            num_inputs *= self.mpu.get_data_parallel_world_size()
        metrics = det.util.make_metrics(num_inputs, per_batch_metrics)

        # Ignore batch_metrics entirely for custom reducers; there's no guarantee that per-batch
        # metrics are even logical for a custom reducer.
        with self.prof.record_timing("reduce_metrics"):
            metrics["avg_metrics"].update(
                self._convert_metrics_to_numpy(self.context.reduce_metrics(for_training=True))
            )

        if not self.is_chief:
            # The training metrics are reported only in the chief process.
            return {}

        logging.debug(f"Done training step: {num_inputs} records in {num_batches} batches.")
        self.prof.set_training(False)

        return metrics

    @staticmethod
    def _convert_metrics_to_numpy(metrics: Dict[str, Any]) -> Dict[str, Any]:
        for metric_name, metric_val in metrics.items():
            if isinstance(metric_val, torch.Tensor):
                metrics[metric_name] = metric_val.cpu().numpy()
        return metrics

    @torch.no_grad()  # type: ignore
    def _compute_validation_metrics(self) -> workload.Response:
        self.context.reset_reducers()
        # Set the behavior of certain layers (e.g., dropout) that are
        # different between training and inference.
        for model in self.context.models:
            model.eval()

        for callback in self.callbacks.values():
            if util.is_overridden(callback.on_validation_step_start, pytorch.PyTorchCallback):
                logging.warning(
                    "on_validation_step_start is now deprecated, "
                    "please use on_validation_start instead"
                )
                callback.on_validation_step_start()

        for callback in self.callbacks.values():
            callback.on_validation_start()

        num_inputs = 0
        metrics = {}  # type: Dict[str, Any]

        if self._evaluate_batch_defined():
            keys = None
            batch_metrics = []

            self.validation_iterator = (
                iter(self.validation_loader) if self.validation_loader is not None else None
            )

            if self.validation_loader is not None:
                self.num_validation_batches = len(self.validation_loader)
                check.gt(self.num_validation_batches, 0)

            # TODO (liam): really we need to broadcast the validation_loader length to all
            # ranks in the same model parallel group.
            all_num_val_batches = self.context.distributed._zmq_gather(self.num_validation_batches)
            if self.is_chief:
                self.num_validation_batches = min(
                    [le for le in all_num_val_batches if le is not None]
                )
            self.num_validation_batches = cast(
                int, self.context.distributed._zmq_broadcast(self.num_validation_batches)
            )

            for callback in self.callbacks.values():
                callback.on_validation_epoch_start()

            for idx in range(self.num_validation_batches):
                if self.validation_iterator is not None:
                    batch = next(self.validation_iterator)
                    if self.context.experimental._auto_to_device:
                        batch = self.context.to_device(batch)
                else:
                    batch = None
                batch_inputs = self.trial.get_batch_length(batch) if batch is not None else None

                if self.is_chief:
                    batch_inputs = cast(int, batch_inputs)
                    assert batch_inputs > 0, "Batch must be non-empty on chief GPU."
                batch_inputs = cast(int, self.context.distributed._zmq_broadcast(batch_inputs))
                num_inputs += batch_inputs

                if has_param(self.trial.evaluate_batch, "batch_idx", 2):
                    vld_metrics = self.trial.evaluate_batch(batch=batch, batch_idx=idx)
                else:
                    vld_metrics = self.trial.evaluate_batch(batch=batch)  # type: ignore
                # Verify validation metric names are the same across batches.
                if self.mpu.should_report_metrics():
                    if keys is None:
                        keys = vld_metrics.keys()
                    else:
                        check.eq(
                            keys,
                            vld_metrics.keys(),
                            "Validation metric names must match across all batches of data.",
                        )
                    check.is_instance(
                        vld_metrics,
                        dict,
                        "validation_metrics() must return a "
                        "dictionary of string names to Tensor "
                        "metrics",
                    )
                    # TODO: For performance perform -> cpu() only at the end of validation.
                    batch_metrics.append(self._convert_metrics_to_numpy(vld_metrics))
                    if self.env.test_mode:
                        break
            # TODO(liam): gather by model parallel group.
            all_keys = self.context.distributed._zmq_gather(keys if keys is None else list(keys))
            if self.is_chief:
                all_keys = [k for k in all_keys if k is not None]
                keys = all_keys[0]
            keys = self.context.distributed._zmq_broadcast(keys)

            for callback in self.callbacks.values():
                callback.on_validation_epoch_end(batch_metrics)

            metrics = self._reduce_metrics(
                batch_metrics=batch_metrics,
                keys=keys,
                metrics_reducers=self._prepare_metrics_reducers(keys=keys),
            )

            if self.mpu.get_data_parallel_world_size() > 1:
                num_inputs *= self.mpu.get_data_parallel_world_size()

            if self.validation_iterator is not None:
                del self.validation_iterator

        else:
            check.true(self._evaluate_full_dataset_defined())
            if self.is_chief:
                self.validation_loader = cast(torch.utils.data.DataLoader, self.validation_loader)
                num_inputs = self.context.get_per_slot_batch_size() * len(self.validation_loader)
            if self.mpu.get_data_parallel_rank() == 0:
                metrics = self.trial.evaluate_full_dataset(data_loader=self.validation_loader)

                check.is_instance(
                    metrics, dict, f"eval() must return a dictionary, got {type(metrics)}."
                )

                metrics = self._convert_metrics_to_numpy(metrics)

        metrics.update(
            self._convert_metrics_to_numpy(self.context.reduce_metrics(for_training=False))
        )

        if self.hvd_config.use and any(
            map(
                lambda c: util.is_overridden(c.on_validation_end, pytorch.PyTorchCallback)
                or util.is_overridden(c.on_validation_step_end, pytorch.PyTorchCallback),
                self.callbacks.values(),
            )
        ):
            logging.debug(
                "Broadcasting metrics to all worker processes to execute a "
                "validation step end callback"
            )
            metrics = hvd.broadcast_object(metrics, root_rank=0)

        for callback in self.callbacks.values():
            if util.is_overridden(callback.on_validation_step_end, pytorch.PyTorchCallback):
                logging.warning(
                    "on_validation_step_end is now deprecated, please use on_validation_end instead"
                )
                callback.on_validation_step_end(metrics)

        for callback in self.callbacks.values():
            callback.on_validation_end(metrics)

        if not self.is_chief:
            return {}

        return {"num_inputs": num_inputs, "validation_metrics": metrics}

    def _prepare_metrics_reducers(self, keys: Any) -> Dict[str, pytorch.Reducer]:
        metrics_reducers = {}  # type: Dict[str, pytorch.Reducer]

        reducer = self.trial.evaluation_reducer()
        if isinstance(reducer, Dict):
            metrics_reducers = reducer
            check.eq(
                list(metrics_reducers.keys()),
                list(keys),
                "Please provide a single evaluation reducer or "
                "provide a reducer for every validation metric. "
                f"Expected keys: {list(keys)}, provided keys: {list(metrics_reducers.keys())}.",
            )
        elif isinstance(reducer, pytorch.Reducer):
            for key in keys:
                metrics_reducers[key] = reducer

        for key in keys:
            check.true(
                isinstance(metrics_reducers[key], pytorch.Reducer),
                "Please select `determined.pytorch.Reducer` for reducing validation metrics.",
            )

        return metrics_reducers

    def _reduce_metrics(
        self, batch_metrics: List, keys: Any, metrics_reducers: Dict[str, pytorch.Reducer]
    ) -> Dict[str, Any]:
        metrics = {}
        if self.mpu.should_report_metrics():
            metrics = {
                name: pytorch._reduce_metrics(
                    reducer=metrics_reducers[name],
                    metrics=np.stack([b[name] for b in batch_metrics], axis=0),
                    num_batches=None,
                )
                for name in keys or []
            }

        if self.context.distributed.get_size() > 1:
            # If using horovod combine metrics across all processes.
            # Only the chief process will receive all the metrics.
            self.validation_loader = cast(torch.utils.data.DataLoader, self.validation_loader)
            num_batches = cast(int, self.num_validation_batches)
            combined_metrics, batches_per_process = self._combine_metrics_across_processes(
                metrics, num_batches
            )
            if self.is_chief:
                # Only the chief collects all the metrics.
                combined_metrics = self._convert_metrics_to_numpy(
                    cast(Dict[str, Any], combined_metrics)
                )
                metrics = {
                    name: pytorch._reduce_metrics(
                        reducer=metrics_reducers[name],
                        metrics=combined_metrics[name],
                        num_batches=batches_per_process,
                    )
                    for name in keys or []
                }
            else:
                return {}

        return metrics

    def _combine_metrics_across_processes(
        self, metrics: Dict[str, Any], num_batches: int
    ) -> Tuple[Optional[Dict[str, Any]], Optional[List[int]]]:
        # The chief receives the metric from every other training process.
        check.true(self.context.distributed.get_size() > 1)

        # all_args is a list of [(metrics, num_batches), ...] for each worker.
        all_args = self.context.distributed._zmq_gather((metrics, num_batches))

        if not self.is_chief:
            return None, None

        # Remove items without keys in dictionary. These are from intermediate model parallel nodes.
        all_args = [a for a in all_args if len(a[0])]

        # Reshape so e.g. all_metrics = [metrics, metrics, ...].
        all_metrics, all_num_batches = zip(*all_args)

        # convert all_metrics from List[Dict[str, Any]] to Dict[str, List[Any]].
        keys = all_metrics[0].keys()
        metrics_lists = {key: [m[key] for m in all_metrics] for key in keys}

        return metrics_lists, all_num_batches

    def _load(self, load_path: pathlib.Path) -> None:
        self.trial.load(self.context, self.callbacks, load_path)

        # Load workload sequencer state.
        wlsq_path = load_path.joinpath("workload_sequencer.pkl")
        if self.wlsq is not None and wlsq_path.exists():
            with wlsq_path.open("rb") as f:
                self.wlsq.load_state(pickle.load(f))

    def _save(self, path: pathlib.Path) -> None:
        if self.context.distributed._is_local_chief:
            path.mkdir(parents=True, exist_ok=True)
        self.context.distributed._zmq_gather("Finished creating checkpoint dir in all containers.")

        if self.is_chief:
            util.write_user_code(path, self.env.on_cluster)

            if self.wlsq is not None:
                with path.joinpath("workload_sequencer.pkl").open("wb") as f:
                    pickle.dump(self.wlsq.get_state(), f)

        self.trial.save(self.context, self.callbacks, path)

    def _sync_device(self) -> None:
        torch.cuda.synchronize(self.context.device)


class PyTorchTrial(det.Trial):
    """
    PyTorch trials are created by subclassing this abstract class.

    We can do the following things in this trial class:

    * **Define models, optimizers, and LR schedulers**.

       In the :meth:`__init__` method, initialize models, optimizers, and LR schedulers
       and wrap them with ``wrap_model``, ``wrap_optimizer``, ``wrap_lr_scheduler``
       provided by :class:`~determined.pytorch.PyTorchTrialContext`.

    * **Run forward and backward passes**.

       In :meth:`train_batch`, call ``backward`` and ``step_optimizer`` provided by
       :class:`~determined.pytorch.PyTorchTrialContext`.
       We support arbitrary numbers of models, optimizers, and LR schedulers
       and arbitrary orders of running forward and backward passes.

    * **Configure automatic mixed precision**.

       In the :meth:`__init__` method, call ``configure_apex_amp`` provided by
       :class:`~determined.pytorch.PyTorchTrialContext`.

    * **Clip gradients**.

       In :meth:`train_batch`, pass a function into
       ``step_optimizer(optimizer, clip_grads=...)`` provided by
       :class:`~determined.pytorch.PyTorchTrialContext`.
    """

    trial_controller_class = PyTorchTrialController
    trial_context_class = pytorch.PyTorchTrialContext

    @abstractmethod
    def __init__(self, context: pytorch.PyTorchTrialContext) -> None:
        """
        Initializes a trial using the provided ``context``. The general steps are:

        1. Initialize model(s) and wrap them with ``context.wrap_model``.
        2. Initialize optimizer(s) and wrap them with ``context.wrap_optimizer``.
        3. Initialize learning rate schedulers and wrap them with ``context.wrap_lr_scheduler``.
        4. If desired, wrap models and optimizer with ``context.configure_apex_amp``
           to use ``apex.amp`` for automatic mixed precision.

        Here is a code example.

        .. code-block:: python

            self.context = context

            self.a = self.context.wrap_model(MyModelA())
            self.b = self.context.wrap_model(MyModelB())
            self.opt1 = self.context.wrap_optimizer(torch.optm.Adam(self.a))
            self.opt2 = self.context.wrap_optimizer(torch.optm.Adam(self.b))

            (self.a, self.b), (self.opt1, self.opt2) = self.context.configure_apex_amp(
                models=[self.a, self.b],
                optimizers=[self.opt1, self.opt2],
                num_losses=2,
            )

            self.lrs1 = self.context.wrap_lr_scheduler(
                lr_scheduler=LambdaLR(self.opt1, lr_lambda=lambda epoch: 0.95 ** epoch),
                step_mode=LRScheduler.StepMode.STEP_EVERY_EPOCH,
            ))
        """
        pass

    @abstractmethod
    def train_batch(
        self, batch: pytorch.TorchData, epoch_idx: int, batch_idx: int
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Train on one batch.

        Users should implement this function by doing the following things:

        1. Run forward passes on the models.

        2. Calculate the gradients with the losses with ``context.backward``.

        3. Call an optimization step for the optimizers with ``context.step_optimizer``.
           You can clip gradients by specifying the argument ``clip_grads``.

        4. Step LR schedulers if using manual step mode.

        5. Return training metrics in a dictionary.

        Here is a code example.

        .. code-block:: python

            # Assume two models, two optimizers, and two LR schedulers were initialized
            # in ``__init__``.

            # Calculate the losses using the models.
            loss1 = self.model1(batch)
            loss2 = self.model2(batch)

            # Run backward passes on losses and step optimizers. These can happen
            # in arbitrary orders.
            self.context.backward(loss1)
            self.context.backward(loss2)
            self.context.step_optimizer(
                self.opt1,
                clip_grads=lambda params: torch.nn.utils.clip_grad_norm_(params, 0.0001),
            )
            self.context.step_optimizer(self.opt2)

            # Step the learning rate.
            self.lrs1.step()
            self.lrs2.step()

            return {"loss1": loss1, "loss2": loss2}

        Arguments:
            batch (Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor):
                batch of data for training.
            epoch_idx (integer): index of the current epoch among all the batches processed
                per device (slot) since the start of training.
            batch_idx (integer): index of the current batch among all the epoches processed
                per device (slot) since the start of training.
        Returns:
            torch.Tensor or Dict[str, Any]:
                training metrics to return.
        """
        pass

    @abstractmethod
    def build_training_data_loader(self) -> pytorch.DataLoader:
        """
        Defines the data loader to use during training.

        Must return an instance of :py:class:`determined.pytorch.DataLoader`.
        """
        pass

    @abstractmethod
    def build_validation_data_loader(self) -> pytorch.DataLoader:
        """
        Defines the data loader to use during validation.

        Must return an instance of :py:class:`determined.pytorch.DataLoader`.
        """
        pass

    def build_callbacks(self) -> Dict[str, pytorch.PyTorchCallback]:
        """
        Defines a dictionary of string names to callbacks to be used during
        training and/or validation.

        The string name will be used as the key to save and restore callback
        state for any callback that defines :meth:`load_state_dict` and :meth:`state_dict`.
        """
        return {}

    def evaluate_batch(self, batch: pytorch.TorchData, batch_idx: int) -> Dict[str, Any]:
        """
        Calculate validation metrics for a batch and return them as a
        dictionary mapping metric names to metric values. Per-batch validation metrics
        are reduced (aggregated) to produce a single set of validation metrics for the
        entire validation set (see :meth:`evaluation_reducer`).

        There are two ways to specify evaluation metrics. Either override
        :meth:`evaluate_batch` or :meth:`evaluate_full_dataset`. While
        :meth:`evaluate_full_dataset` is more flexible,
        :meth:`evaluate_batch` should be preferred, since it can be
        parallelized in distributed environments, whereas
        :meth:`evaluate_full_dataset` cannot. Only one of
        :meth:`evaluate_full_dataset` and :meth:`evaluate_batch` should be
        overridden by a trial.

        The metrics returned from this function must be JSON-serializable.

        Arguments:
            batch (Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor):
                batch of data for evaluating.
        """
        pass

    def evaluation_reducer(self) -> Union[pytorch.Reducer, Dict[str, pytorch.Reducer]]:
        """
        Return a reducer for all evaluation metrics, or a dict mapping metric
        names to individual reducers. Defaults to :obj:`determined.pytorch.Reducer.AVG`.
        """
        return pytorch.Reducer.AVG

    def evaluate_full_dataset(
        self, data_loader: Optional[torch.utils.data.DataLoader]
    ) -> Dict[str, Any]:
        """
        Calculate validation metrics on the entire validation dataset and
        return them as a dictionary mapping metric names to reduced metric
        values (i.e., each returned metric is the average or sum of that metric
        across the entire validation set).

        This validation cannot be distributed and is performed on a single
        device, even when multiple devices (slots) are used for training. Only
        one of :meth:`evaluate_full_dataset` and :meth:`evaluate_batch` should
        be overridden by a trial.

        The metrics returned from this function must be JSON-serializable.

        Arguments:
            data_loader (torch.utils.data.DataLoader): data loader for evaluating.
        """
        pass

    def get_batch_length(self, batch: Any) -> int:
        """Count the number of records in a given batch.

        Override this method when you are using custom batch types, as produced
        when iterating over the `DataLoader`.
        For example, when using `pytorch_geometric`:

        .. code-block:: python

            # Extra imports:
            from determined.pytorch import DataLoader
            from torch_geometric.data.dataloader import Collater

            # Trial methods:
            def build_training_data_loader(self):
                return DataLoader(
                    self.train_subset,
                    batch_size=self.context.get_per_slot_batch_size(),
                    collate_fn=Collater([], []),
                )

            def get_batch_length(self, batch):
                # `batch` is `torch_geometric.data.batch.Batch`.
                return batch.num_graphs

        Arguments:
            batch (Any): input training or validation data batch object.
        """
        return pytorch.data_length(batch)

    def save(
        self,
        context: pytorch.PyTorchTrialContext,
        callbacks: Dict[str, pytorch.PyTorchCallback],
        path: pathlib.Path,
    ) -> None:
        if not context.distributed._is_chief:
            return
        rng_state = {
            "cpu_rng_state": torch.random.get_rng_state(),
            "np_rng_state": np.random.get_state(),
            "random_rng_state": random.getstate(),
        }

        if torch.cuda.device_count():
            rng_state["gpu_rng_state"] = torch.cuda.get_rng_state(
                context.distributed.get_local_rank()
            )

        # PyTorch uses optimizer objects that take the model parameters to
        # optimize on construction, so we store and reload the `state_dict()`
        # of the model and optimizer explicitly (instead of dumping the entire
        # objects) to avoid breaking the connection between the model and the
        # optimizer.
        checkpoint = {
            "models_state_dict": [model.state_dict() for model in context.models],
            "optimizers_state_dict": [optimizer.state_dict() for optimizer in context.optimizers],
            "lr_schedulers_state_dict": [
                lr_scheduler.state_dict() for lr_scheduler in context.lr_schedulers
            ],
            "rng_state": rng_state,
        }

        if context._scaler:
            checkpoint["scaler_state_dict"] = context._scaler.state_dict()

        if context._use_apex:
            checkpoint["amp_state"] = apex.amp.state_dict()

        checkpoint["callbacks"] = {
            name: callback.state_dict() for name, callback in callbacks.items()
        }

        for callback in callbacks.values():
            callback.on_checkpoint_save_start(checkpoint)

        ckpt_name = "state_dict.pth"
        torch.save(checkpoint, str(path.joinpath(ckpt_name)))

        for callback in callbacks.values():
            callback.on_checkpoint_end(str(path))

    def load(
        self,
        context: pytorch.PyTorchTrialContext,
        callbacks: Dict[str, pytorch.PyTorchCallback],
        load_path: pathlib.Path,
    ) -> None:
        # Backwards compat with older checkpoint formats. List is newest to
        # oldest known state_dict locations.
        potential_paths = [
            ["state_dict.pth"],
            ["determined", "state_dict.pth"],
            ["pedl", "state_dict.pth"],
            ["checkpoint.pt"],
        ]

        checkpoint: Optional[Dict[str, Any]] = None
        for ckpt_path in potential_paths:
            maybe_ckpt = load_path.joinpath(*ckpt_path)
            if maybe_ckpt.exists():
                checkpoint = torch.load(str(maybe_ckpt), map_location="cpu")  # type: ignore
                break
        if checkpoint is None or not isinstance(checkpoint, dict):
            return

        for callback in callbacks.values():
            callback.on_checkpoint_load_start(checkpoint)

        if "model_state_dict" in checkpoint:
            # Backward compatible with older checkpoint format.
            check.not_in("models_state_dict", checkpoint)
            check.eq(len(context.models), 1)
            context.models[0].load_state_dict(checkpoint["model_state_dict"])
        else:
            for idx, model in enumerate(context.models):
                model.load_state_dict(checkpoint["models_state_dict"][idx])

        if "optimizer_state_dict" in checkpoint:
            # Backward compatible with older checkpoint format.
            check.not_in("optimizers_state_dict", checkpoint)
            check.eq(len(context.optimizers), 1)
            context.optimizers[0].load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            for idx, optimizer in enumerate(context.optimizers):
                optimizer.load_state_dict(checkpoint["optimizers_state_dict"][idx])

        if "lr_scheduler" in checkpoint:
            # Backward compatible with older checkpoint format.
            check.not_in("lr_schedulers_state_dict", checkpoint)
            check.eq(len(context.lr_schedulers), 1)
            context.lr_schedulers[0].load_state_dict(checkpoint["lr_scheduler"])
        else:
            for idx, lr_scheduler in enumerate(context.lr_schedulers):
                lr_scheduler.load_state_dict(checkpoint["lr_schedulers_state_dict"][idx])

        if "scaler_state_dict" in checkpoint:
            if context._scaler:
                context._scaler.load_state_dict(checkpoint["scaler_state_dict"])
            else:
                logging.warning(
                    "There exists scaler_state_dict in checkpoint but the experiment is not using "
                    "AMP."
                )
        else:
            if context._scaler:
                logging.warning(
                    "The experiment is using AMP but scaler_state_dict does not exist in the "
                    "checkpoint."
                )

        if "amp_state" in checkpoint:
            if context._use_apex:
                apex.amp.load_state_dict(checkpoint["amp_state"])
            else:
                logging.warning(
                    "There exists amp_state in checkpoint but the experiment is not using Apex."
                )
        else:
            if context._use_apex:
                logging.warning(
                    "The experiment is using Apex but amp_state does not exist in the checkpoint."
                )

        if "rng_state" in checkpoint:
            rng_state = checkpoint["rng_state"]
            np.random.set_state(rng_state["np_rng_state"])
            random.setstate(rng_state["random_rng_state"])
            torch.random.set_rng_state(rng_state["cpu_rng_state"])

            if torch.cuda.device_count():
                if "gpu_rng_state" in rng_state:
                    torch.cuda.set_rng_state(
                        rng_state["gpu_rng_state"], device=context.distributed.get_local_rank()
                    )
                else:
                    logging.warning(
                        "The system has a gpu but no gpu_rng_state exists in the checkpoint."
                    )
            else:
                if "gpu_rng_state" in rng_state:
                    logging.warning(
                        "There exists gpu_rng_state in checkpoint but the system has no gpu."
                    )
        else:
            logging.warning("The checkpoint has no random state to restore.")

        callback_state = checkpoint.get("callbacks", {})
        for name in callbacks:
            if name in callback_state:
                callbacks[name].load_state_dict(callback_state[name])
            elif util.is_overridden(callbacks[name].load_state_dict, pytorch.PyTorchCallback):
                logging.warning(
                    "Callback '{}' implements load_state_dict(), but no callback state "
                    "was found for that name when restoring from checkpoint. This "
                    "callback will be initialized from scratch"
                )


def reset_parameters(model: torch.nn.Module) -> None:
    """
    .. warning::
        ``det.pytorch.reset_parameters()`` is deprecated and should not be called. For custom
        nn.Modules which do need a call to reset_parameters(), it is recommended to call
        self.reset_parameters() directly in their __init__() function, as is standard in all
        built-in nn.Modules.

    Recursively calls ``reset_parameters()`` for all modules.
    """
    logging.warning(
        "det.pytorch.reset_parameters() is deprecated and should not be called.  For custom "
        "nn.Modules which do need a call to reset_parameters(), it is recommended to call "
        "self.reset_parameters() directly in their __init__() function, as is standard in all "
        "built-in nn.Modules."
    )
    for _, module in model.named_modules():
        reset_params = getattr(module, "reset_parameters", None)
        if callable(reset_params):
            reset_params()
