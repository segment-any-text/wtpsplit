from transformers import TrainingArguments
from transformers.training_args import ParallelMode
from transformers.utils import (
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_available,
    is_torch_tpu_available,
    requires_backends,
)
import numbers
import os


from transformers.utils import logging
from transformers.integrations import (
    rewrite_logs,
    WandbCallback,
    AzureMLCallback,
    CometCallback,
    MLflowCallback,
    NeptuneCallback,
    TensorBoardCallback,
    CodeCarbonCallback,
    ClearMLCallback,
    DagsHubCallback,
)

logger = logging.get_logger(__name__)
if is_torch_available():
    import torch
    import torch.distributed as dist

if is_sagemaker_mp_enabled():
    import smp.distributed.modelparallel.torch as smp

    smp.init()


class ParallelTPUAdapterTrainingArguments(TrainingArguments):
    """
    Subclass of `TrainingArguments`, specific to training on TPU VMs in parallel using different data.
    (Different optimization on different TPU cores, different data on different TPU cores, etc.)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def world_size(self):
        """
        The number of processes used in parallel.
        """
        requires_backends(self, ["torch"])

        if is_torch_tpu_available():
            # MODIFIED: otherwise, Trainer only logs on main (0) process, and DataLoader is of distributed type
            return 1
        elif is_sagemaker_mp_enabled():
            return smp.dp_size() if not smp.state.cfg.prescaled_batch else smp.rdp_size()
        elif is_sagemaker_dp_enabled():
            return dist.get_world_size()
        elif self.parallel_mode == ParallelMode.DISTRIBUTED:
            return torch.distributed.get_world_size()
        return 1


class ParallelTPUWandbCallback(WandbCallback):
    """
    A [`TrainerCallback`] that logs metrics, media, model checkpoints to [Weight and Biases](https://www.wandb.com/).
    """

    def __init__(self):
        super().__init__()

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        # MODIFIED: log on all processes
        # if state.is_world_process_zero:
        logs = rewrite_logs(logs)
        self._wandb.log({**logs, "train/global_step": state.global_step})

    def on_save(self, args, state, control, **kwargs):
        # MODIFIED: save on all
        if self._log_model == "checkpoint" and self._initialized:
            checkpoint_metadata = {
                k: v
                for k, v in dict(self._wandb.summary).items()
                if isinstance(v, numbers.Number) and not k.startswith("_")
            }

            ckpt_dir = f"checkpoint-{state.global_step}"
            artifact_path = os.path.join(args.output_dir, ckpt_dir)
            logger.info(f"Logging checkpoint artifacts in {ckpt_dir}. ...")
            checkpoint_name = (
                f"checkpoint-{self._wandb.run.id}"
                if (args.run_name is None or args.run_name == args.output_dir)
                else f"checkpoint-{self._wandb.run.name}"
            )
            artifact = self._wandb.Artifact(name=checkpoint_name, type="model", metadata=checkpoint_metadata)
            artifact.add_dir(artifact_path)
            self._wandb.log_artifact(artifact, aliases=[f"checkpoint-{state.global_step}"])


INTEGRATION_TO_CALLBACK = {
    "azure_ml": AzureMLCallback,
    "comet_ml": CometCallback,
    "mlflow": MLflowCallback,
    "neptune": NeptuneCallback,
    "tensorboard": TensorBoardCallback,
    "wandb": ParallelTPUWandbCallback,
    "codecarbon": CodeCarbonCallback,
    "clearml": ClearMLCallback,
    "dagshub": DagsHubCallback,
}