import abc
import logging
from typing import Any, Dict

import determined as det
from determined import _generic, horovod


class TrialContext(metaclass=abc.ABCMeta):
    """
    TrialContext is the system-provided API to a Trial class.
    """

    def __init__(
        self,
        generic_context: _generic.Context,
        env: det.EnvContext,
        hvd_config: horovod.HorovodContext,
    ) -> None:
        self._generic = generic_context
        self.env = env
        self.hvd_config = hvd_config

        self.distributed = self._generic.distributed
        self._stop_requested = False

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TrialContext":
        """
        Create an context object suitable for debugging outside of Determined.

        An example for a subclass of :class:`~determined.pytorch._pytorch_trial.PyTorchTrial`:

        .. code-block:: python

            config = { ... }
            context = det.pytorch.PyTorchTrialContext.from_config(config)
            my_trial = MyPyTorchTrial(context)

            train_ds = my_trial.build_training_data_loader()
            for epoch_idx in range(3):
                for batch_idx, batch in enumerate(train_ds):
                    metrics = my_trial.train_batch(batch, epoch_idx, batch_idx)
                    ...

        An example for a subclass of :class:`~determined.keras._tf_keras_trial.TFKerasTrial`:

        .. code-block:: python

            config = { ... }
            context = det.keras.TFKerasTrialContext.from_config(config)
            my_trial = tf_keras_one_var_model.OneVarTrial(context)

            model = my_trial.build_model()
            model.fit(my_trial.build_training_data_loader())
            eval_metrics = model.evaluate(my_trial.build_validation_data_loader())

        Arguments:
            config: An experiment config file, in dictionary form.
        """
        generic_context, env, hvd_config = det._make_local_execution_env(
            managed_training=False,
            test_mode=False,
            config=config,
            checkpoint_dir="/tmp",
            limit_gpus=1,
        )
        return cls(generic_context, env, hvd_config)

    def get_experiment_config(self) -> Dict[str, Any]:
        """
        Return the experiment configuration.
        """
        return self.env.experiment_config

    def get_data_config(self) -> Dict[str, Any]:
        """
        Return the data configuration.
        """
        return self.get_experiment_config().get("data", {})

    def get_experiment_id(self) -> int:
        """
        Return the experiment ID of the current trial.
        """
        return int(self.env.det_experiment_id)

    def get_global_batch_size(self) -> int:
        """
        Return the global batch size.
        """
        return self.env.global_batch_size

    def get_per_slot_batch_size(self) -> int:
        """
        Return the per-slot batch size. When a model is trained with a single GPU, this is equal to
        the global batch size. When multi-GPU training is used, this is equal to the global batch
        size divided by the number of GPUs used to train the model.
        """
        return self.env.per_slot_batch_size

    def get_trial_id(self) -> int:
        """
        Return the trial ID of the current trial.
        """
        return int(self.env.det_trial_id)

    def get_trial_seed(self) -> int:
        return self.env.trial_seed

    def get_hparams(self) -> Dict[str, Any]:
        """
        Return a dictionary of hyperparameter names to values.
        """
        return self.env.hparams

    def get_hparam(self, name: str) -> Any:
        """
        Return the current value of the hyperparameter with the given name.
        """
        if name not in self.env.hparams:
            raise ValueError(
                "Could not find name '{}' in experiment "
                "hyperparameters. Please check your experiment "
                "configuration 'hyperparameters' section.".format(name)
            )
        if name == "global_batch_size":
            logging.warning(
                "Please use `context.get_per_slot_batch_size()` and "
                "`context.get_global_batch_size()` instead of accessing "
                "`global_batch_size` directly."
            )
        return self.env.hparams[name]

    def get_stop_requested(self) -> bool:
        """
        Return whether a trial stoppage has been requested.
        """
        return self._stop_requested

    def set_stop_requested(self, stop_requested: bool) -> None:
        """
        Set a flag to request a trial stoppage. When this flag is set to True,
        we finish the step, checkpoint, then exit.
        """
        if not isinstance(stop_requested, bool):
            raise AssertionError("stop_requested must be a boolean")

        logging.info(
            "A trial stoppage has requested. The trial will be stopped "
            "at the end of the current step."
        )
        self._stop_requested = stop_requested

    def get_initial_batch(self) -> int:
        return self.env.latest_batch
