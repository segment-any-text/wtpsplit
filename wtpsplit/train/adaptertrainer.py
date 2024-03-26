import math
import os
import shutil
import sys
import time
from typing import Dict

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

# Integrations must be imported before ML frameworks:

# isort: off
from transformers.integrations import (
    hp_params,
)

# isort: on

from transformers import PreTrainedModel
from transformers.modeling_utils import unwrap_model
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    WEIGHTS_NAME,
    DataLoader,
    EvalLoopOutput,
    IterableDatasetShard,
    List,
    Optional,
    deepspeed_init,
    denumpify_detensorize,
    find_batch_size,
    get_parameter_names,
    has_length,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    nested_concat,
    nested_numpify,
    nested_truncate,
)
from transformers.trainer_utils import TrainOutput, speed_metrics

from wtpsplit.train.utils import Model

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm  # noqa: F401
    import torch_xla.debug.metrics as met  # noqa: F401
    import torch_xla.distributed.parallel_loader as pl  # noqa: F401

import re
from typing import Callable, Tuple, Union

import torch.distributed as dist
from packaging import version
from torch.utils.data import Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import Trainer, __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.pytorch_utils import is_torch_less_than_1_11
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import TRAINER_STATE_NAME
from transformers.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    get_model_param_count,
)
from transformers.trainer_utils import (
    EvalPrediction,
    HPSearchBackend,
    ShardedDDPOption,
)
from transformers.training_args import ParallelMode, TrainingArguments
from transformers.utils import (
    CONFIG_NAME,
    is_accelerate_available,
    is_apex_available,
    is_torch_tpu_available,
    logging,
)

from adapters.composition import AdapterCompositionBlock, Fuse

if is_apex_available():
    from apex import amp

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

skip_first_batches = None
if is_accelerate_available():
    from accelerate import __version__ as accelerate_version

    if version.parse(accelerate_version) >= version.parse("0.16"):
        from accelerate import skip_first_batches
logger = logging.get_logger(__name__)


class AdapterTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        adapter_names: Optional[List[List[str]]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        logging_prefix="",
        skip_eval_loss: bool = False,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=[AdapterTrainerCallback(self)] + callbacks if callbacks else [AdapterTrainerCallback(self)],
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        self.logging_prefix = logging_prefix
        self.skip_eval_loss = skip_eval_loss

        if adapter_names is not None:
            self.model.backbone.set_active_adapters(adapter_names)
        # Set the defaults for loading/ saving model & adapters
        if isinstance(self.model.backbone, PreTrainedModel):
            model_frozen = getattr(self.model.backbone.base_model, "model_frozen", False)
        else:
            model_frozen = False
        if model_frozen and self.model.backbone.active_adapters:
            # Check if training AdapterFusion
            self.train_adapter_fusion = (
                isinstance(self.model.backbone.active_adapters, Fuse)
                or isinstance(self.model.backbone.active_adapters, AdapterCompositionBlock)
                and any([isinstance(child, Fuse) for child in self.model.backbone.active_adapters.children])
            )
        if self.model.backbone.active_adapters is None:
            raise ValueError(
                "Expected a model with an active adapter setup."
                "If you want to fully finetune the model use the Trainer class."
            )
        if (self.label_names is None or len(self.label_names) < 1) and self.model.active_head is not None:
            all_label_names = set()
            for head in self.model.backbone._active_heads:
                all_label_names |= set(self.model.backbone.heads[head].get_label_names())
            self.label_names = list(all_label_names)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if hasattr(self.model, "config") and hasattr(self.model.config, "adapters"):
                match_str = r"adapter_fusion_layer\..*\.value"
                decay_parameters = [name for name in decay_parameters if not re.match(match_str, name)]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                if state_dict is None:
                    state_dict = self.model.state_dict()
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_all_adapters(output_dir)
            if self.train_adapter_fusion:
                self.model.save_all_adapter_fusions(output_dir)
            if hasattr(self.model, "heads"):
                self.model.save_all_heads(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _load_from_checkpoint(self, resume_from_checkpoint):
        args = self.args
        if os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
            logger.info(f"Loading model from {resume_from_checkpoint}).")

        if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
            config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
            checkpoint_version = config.transformers_version
            if checkpoint_version is not None and checkpoint_version != __version__:
                logger.warn(
                    f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                    f"Transformers but your current version is {__version__}. This is not recommended and could "
                    "yield to errors or unwanted behaviors."
                )

        if args.deepspeed:
            # will be resumed in deepspeed_init
            pass
        else:
            adapter_loaded = False
            if os.path.isdir(resume_from_checkpoint):
                adapter_loaded = self._load_adapters(resume_from_checkpoint)
                self._load_adapter_fusions(resume_from_checkpoint)
                # Save all heads for a model with heads
                if hasattr(self.model, "heads"):
                    self._load_heads(resume_from_checkpoint)

            if not adapter_loaded:
                raise Exception("Can't find a valid checkpoint at {}".format(resume_from_checkpoint))

    def _load_adapters(self, resume_from_checkpoint):
        adapter_loaded = False
        for file_name in os.listdir(resume_from_checkpoint):
            if os.path.isdir(os.path.join(resume_from_checkpoint, file_name)):
                if "," not in file_name and "adapter_config.json" in os.listdir(
                    os.path.join(resume_from_checkpoint, file_name)
                ):
                    self.model.load_adapter(os.path.join(os.path.join(resume_from_checkpoint, file_name)))
                    adapter_loaded = True
        return adapter_loaded

    def _load_adapter_fusions(self, resume_from_checkpoint):
        for file_name in os.listdir(resume_from_checkpoint):
            if os.path.isdir(os.path.join(resume_from_checkpoint, file_name)):
                if "," in file_name:
                    self.model.load_adapter_fusion(os.path.join(resume_from_checkpoint, file_name))

    def _load_heads(self, resume_from_checkpoint):
        for file_name in os.listdir(resume_from_checkpoint):
            if os.path.isdir(os.path.join(resume_from_checkpoint, file_name)):
                if "," not in file_name and "head_config.json" in os.listdir(
                    os.path.join(resume_from_checkpoint, file_name)
                ):
                    self.model.load_head(os.path.join(resume_from_checkpoint, file_name))

    def _load_best_model(self):
        model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        logger.info(
            f"Loading best adapter(s) from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
        )
        # attempt to re-load all adapters from checkpoint
        for adapter in model.adapters_config.adapters:
            adapter_dir = os.path.join(self.state.best_model_checkpoint, adapter)
            if os.path.exists(adapter_dir):
                model.load_adapter(adapter_dir)
        if self.train_adapter_fusion:
            logger.info(
                f"Loading best adapter fusion(s) from {self.state.best_model_checkpoint} (score:"
                f" {self.state.best_metric})."
            )
            # attempt to re-load all adapter fusions from checkpoint
            for fusion in model.adapters_config.fusions:
                fusion_dir = os.path.join(self.state.best_model_checkpoint, fusion)
                if os.path.exists(fusion_dir):
                    model.load_adapter_fusion(fusion_dir)
        model.to(self.args.device)

    # def _inner_training_loop(
    #     self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    # ):
    #     self._train_batch_size = batch_size
    #     # Data loader and number of training steps
    #     train_dataloader = self.get_train_dataloader()

    #     # Setting up training control variables:
    #     # number of training epochs: num_train_epochs
    #     # number of training steps per epoch: num_update_steps_per_epoch
    #     # total number of training steps to execute: max_steps
    #     total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

    #     len_dataloader = None
    #     if has_length(train_dataloader):
    #         len_dataloader = len(train_dataloader)
    #         num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
    #         num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    #         num_examples = self.num_examples(train_dataloader)
    #         if args.max_steps > 0:
    #             max_steps = args.max_steps
    #             num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
    #                 args.max_steps % num_update_steps_per_epoch > 0
    #             )
    #             # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
    #             # the best we can do.
    #             num_train_samples = args.max_steps * total_train_batch_size
    #         else:
    #             max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
    #             num_train_epochs = math.ceil(args.num_train_epochs)
    #             num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
    #     elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
    #         max_steps = args.max_steps
    #         # Setting a very large number of epochs so we go as many times as necessary over the iterator.
    #         num_train_epochs = sys.maxsize
    #         num_update_steps_per_epoch = max_steps
    #         num_examples = total_train_batch_size * args.max_steps
    #         num_train_samples = args.max_steps * total_train_batch_size
    #     else:
    #         raise ValueError(
    #             "args.max_steps must be set to a positive value if dataloader does not have a length, was"
    #             f" {args.max_steps}"
    #         )

    #     # Compute absolute values for logging, eval, and save if given as ratio
    #     if args.logging_steps and args.logging_steps < 1:
    #         args.logging_steps = math.ceil(max_steps * args.logging_steps)
    #     if args.eval_steps and args.eval_steps < 1:
    #         args.eval_steps = math.ceil(max_steps * args.eval_steps)
    #     if args.save_steps and args.save_steps < 1:
    #         args.save_steps = math.ceil(max_steps * args.save_steps)

    #     if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
    #         if self.args.n_gpu > 1:
    #             # nn.DataParallel(model) replicates the model, creating new variables and module
    #             # references registered here no longer work on other gpus, breaking the module
    #             raise ValueError(
    #                 "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
    #                 " (torch.distributed.launch)."
    #             )
    #         else:
    #             debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

    #     delay_optimizer_creation = (
    #         self.sharded_ddp is not None
    #         and self.sharded_ddp != ShardedDDPOption.SIMPLE
    #         or is_sagemaker_mp_enabled()
    #         or self.fsdp is not None
    #     )
    #     if args.deepspeed:
    #         deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
    #             self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
    #         )
    #         self.model = deepspeed_engine.module
    #         self.model_wrapped = deepspeed_engine
    #         self.deepspeed = deepspeed_engine
    #         self.optimizer = optimizer
    #         self.lr_scheduler = lr_scheduler
    #     elif not delay_optimizer_creation:
    #         self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    #     self.state = TrainerState()
    #     self.state.is_hyper_param_search = trial is not None

    #     # Activate gradient checkpointing if needed
    #     if args.gradient_checkpointing:
    #         self.model.gradient_checkpointing_enable()

    #     model = self._wrap_model(self.model_wrapped)

    #     if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
    #         self._load_from_checkpoint(resume_from_checkpoint, model)

    #     # for the rest of this function `model` is the outside model, whether it was wrapped or not
    #     if model is not self.model:
    #         self.model_wrapped = model

    #     if delay_optimizer_creation:
    #         self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    #     # Check if saved optimizer or scheduler states exist
    #     self._load_optimizer_and_scheduler(resume_from_checkpoint)

    #     # important: at this point:
    #     # self.model         is the Transformers Model
    #     # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

    #     # Train!
    #     # MODIFIED: changed to warn, added device
    #     logger.warning(f"***** Running training on {xm.get_ordinal()}*****")
    #     logger.warning(f"  Num examples = {num_examples:,}")
    #     logger.warning(f"  Num Epochs = {num_train_epochs:,}")
    #     logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size:,}")
    #     logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
    #     logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    #     logger.warning(f"  Total optimization steps = {max_steps:,}")
    #     logger.warning(f"  Eval steps = {args.eval_steps}")
    #     logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

    #     self.state.epoch = 0
    #     start_time = time.time()
    #     epochs_trained = 0
    #     steps_trained_in_current_epoch = 0
    #     steps_trained_progress_bar = None

    #     # Check if continuing training from a checkpoint
    #     if resume_from_checkpoint is not None and os.path.isfile(
    #         os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
    #     ):
    #         self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
    #         epochs_trained = self.state.global_step // num_update_steps_per_epoch
    #         if not args.ignore_data_skip:
    #             steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
    #             steps_trained_in_current_epoch *= args.gradient_accumulation_steps
    #         else:
    #             steps_trained_in_current_epoch = 0

    #         logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    #         logger.info(f"  Continuing training from epoch {epochs_trained}")
    #         logger.info(f"  Continuing training from global step {self.state.global_step}")
    #         if not args.ignore_data_skip:
    #             if skip_first_batches is None:
    #                 logger.info(
    #                     f"  Will skip the first {epochs_trained} epochs then the first"
    #                     f" {steps_trained_in_current_epoch} batches in the first epoch. If this takes a lot of time,"
    #                     " you can install the latest version of Accelerate with `pip install -U accelerate`.You can"
    #                     " also add the `--ignore_data_skip` flag to your launch command, but you will resume the"
    #                     " training on data already seen by your model."
    #                 )
    #             else:
    #                 logger.info(
    #                     f"  Will skip the first {epochs_trained} epochs then the first"
    #                     f" {steps_trained_in_current_epoch} batches in the first epoch."
    #                 )
    #             if self.is_local_process_zero() and not args.disable_tqdm and skip_first_batches is None:
    #                 steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
    #                 steps_trained_progress_bar.set_description("Skipping the first batches")

    #     # Update the references
    #     self.callback_handler.model = self.model
    #     self.callback_handler.optimizer = self.optimizer
    #     self.callback_handler.lr_scheduler = self.lr_scheduler
    #     self.callback_handler.train_dataloader = train_dataloader
    #     if self.hp_name is not None and self._trial is not None:
    #         # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
    #         # parameter to Train when using DDP.
    #         self.state.trial_name = self.hp_name(self._trial)
    #     if trial is not None:
    #         assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
    #         self.state.trial_params = hp_params(assignments)
    #     else:
    #         self.state.trial_params = None
    #     # This should be the same if the state has been saved but in case the training arguments changed, it's safer
    #     # to set this after the load.
    #     self.state.max_steps = max_steps
    #     self.state.num_train_epochs = num_train_epochs
    #     self.state.is_local_process_zero = self.is_local_process_zero()
    #     self.state.is_world_process_zero = self.is_world_process_zero()

    #     # tr_loss is a tensor to avoid synchronization of TPUs through .item()
    #     tr_loss = torch.tensor(0.0).to(args.device)
    #     # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
    #     self._total_loss_scalar = 0.0
    #     self._globalstep_last_logged = self.state.global_step
    #     model.zero_grad()

    #     self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

    #     # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
    #     if not args.ignore_data_skip:
    #         for epoch in range(epochs_trained):
    #             is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
    #                 train_dataloader.sampler, RandomSampler
    #             )
    #             if is_torch_less_than_1_11 or not is_random_sampler:
    #                 # We just need to begin an iteration to create the randomization of the sampler.
    #                 # That was before PyTorch 1.11 however...
    #                 for _ in train_dataloader:
    #                     break
    #             else:
    #                 # Otherwise we need to call the whooooole sampler cause there is some random operation added
    #                 # AT THE VERY END!
    #                 _ = list(train_dataloader.sampler)

    #     total_batched_samples = 0
    #     for epoch in range(epochs_trained, num_train_epochs):
    #         if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
    #             train_dataloader.sampler.set_epoch(epoch)
    #         elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
    #             train_dataloader.dataset.set_epoch(epoch)

    #         # if is_torch_tpu_available():
    #         #     parallel_loader = pl.MpDeviceLoader(train_dataloader, args.device)  # .per_device_loader(args.device)
    #         #     epoch_iterator = parallel_loader
    #         # else:
    #         # MODIFIED: no parallel loader
    #         epoch_iterator = train_dataloader

    #         # Reset the past mems state at the beginning of each epoch if necessary.
    #         if args.past_index >= 0:
    #             self._past = None

    #         steps_in_epoch = (
    #             len(epoch_iterator) if len_dataloader is not None else args.max_steps * args.gradient_accumulation_steps
    #         )
    #         self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

    #         if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
    #             self._load_rng_state(resume_from_checkpoint)

    #         rng_to_sync = False
    #         steps_skipped = 0
    #         if skip_first_batches is not None and steps_trained_in_current_epoch > 0:
    #             epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
    #             steps_skipped = steps_trained_in_current_epoch
    #             steps_trained_in_current_epoch = 0
    #             rng_to_sync = True

    #         step = -1
    #         for step, inputs in enumerate(epoch_iterator):
    #             total_batched_samples += 1
    #             if rng_to_sync:
    #                 self._load_rng_state(resume_from_checkpoint)
    #                 rng_to_sync = False

    #             # Skip past any already trained steps if resuming training
    #             if steps_trained_in_current_epoch > 0:
    #                 steps_trained_in_current_epoch -= 1
    #                 if steps_trained_progress_bar is not None:
    #                     steps_trained_progress_bar.update(1)
    #                 if steps_trained_in_current_epoch == 0:
    #                     self._load_rng_state(resume_from_checkpoint)
    #                 continue
    #             elif steps_trained_progress_bar is not None:
    #                 steps_trained_progress_bar.close()
    #                 steps_trained_progress_bar = None

    #             if step % args.gradient_accumulation_steps == 0:
    #                 self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

    #             if (
    #                 (total_batched_samples % args.gradient_accumulation_steps != 0)
    #                 and args.parallel_mode == ParallelMode.DISTRIBUTED
    #                 and args._no_sync_in_gradient_accumulation
    #                 and hasattr(model, "no_sync")
    #             ):
    #                 # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
    #                 with model.no_sync():
    #                     tr_loss_step = self.training_step(model, inputs)
    #             else:
    #                 tr_loss_step = self.training_step(model, inputs)

    #             if (
    #                 args.logging_nan_inf_filter
    #                 and not is_torch_tpu_available()
    #                 and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
    #             ):
    #                 # if loss is nan or inf simply add the average of previous logged losses
    #                 tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
    #             else:
    #                 tr_loss += tr_loss_step

    #             self.current_flos += float(self.floating_point_ops(inputs))

    #             # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
    #             if self.deepspeed:
    #                 self.deepspeed.step()

    #             if total_batched_samples % args.gradient_accumulation_steps == 0 or (
    #                 # last step in epoch but step is always smaller than gradient_accumulation_steps
    #                 steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
    #             ):
    #                 # Gradient clipping
    #                 if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
    #                     # deepspeed does its own clipping

    #                     if self.do_grad_scaling:
    #                         # Reduce gradients first for XLA
    #                         if is_torch_tpu_available():
    #                             gradients = xm._fetch_gradients(self.optimizer)
    #                             xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
    #                         # AMP: gradients need unscaling
    #                         self.scaler.unscale_(self.optimizer)

    #                     if is_sagemaker_mp_enabled() and args.fp16:
    #                         self.optimizer.clip_master_grads(args.max_grad_norm)
    #                     elif hasattr(self.optimizer, "clip_grad_norm"):
    #                         # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
    #                         self.optimizer.clip_grad_norm(args.max_grad_norm)
    #                     elif hasattr(model, "clip_grad_norm_"):
    #                         # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
    #                         model.clip_grad_norm_(args.max_grad_norm)
    #                     else:
    #                         # Revert to normal clipping otherwise, handling Apex or full precision
    #                         nn.utils.clip_grad_norm_(
    #                             amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
    #                             args.max_grad_norm,
    #                         )

    #                 # Optimizer step
    #                 optimizer_was_run = True
    #                 if self.deepspeed:
    #                     pass  # called outside the loop
    #                 elif is_torch_tpu_available():
    #                     if self.do_grad_scaling:
    #                         self.scaler.step(self.optimizer)
    #                         self.scaler.update()
    #                     else:
    #                         # xm.optimizer_step(self.optimizer)
    #                         # MODIFIED: Crucial change! Do not aggregate gradients across TPU cores
    #                         self.optimizer.step()
    #                         xm.mark_step()
    #                 elif self.do_grad_scaling:
    #                     scale_before = self.scaler.get_scale()
    #                     self.scaler.step(self.optimizer)
    #                     self.scaler.update()
    #                     scale_after = self.scaler.get_scale()
    #                     optimizer_was_run = scale_before <= scale_after
    #                 else:
    #                     self.optimizer.step()

    #                 if optimizer_was_run and not self.deepspeed:
    #                     # Delay optimizer scheduling until metrics are generated
    #                     if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
    #                         self.lr_scheduler.step()

    #                 model.zero_grad()
    #                 self.state.global_step += 1
    #                 self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
    #                 self.control = self.callback_handler.on_step_end(args, self.state, self.control)

    #                 self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
    #             else:
    #                 self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

    #             if self.control.should_epoch_stop or self.control.should_training_stop:
    #                 break
    #         if step < 0:
    #             logger.warning(
    #                 "There seems to be not a single sample in your epoch_iterator, stopping training at step"
    #                 f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
    #                 f" num_steps ({max_steps}) higher than the number of available samples."
    #             )
    #             self.control.should_training_stop = True

    #         self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
    #         self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

    #         if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
    #             if is_torch_tpu_available():
    #                 # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
    #                 xm.master_print(met.metrics_report())
    #             else:
    #                 logger.warning(
    #                     "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
    #                     "configured. Check your training configuration if this is unexpected."
    #                 )
    #         if self.control.should_training_stop:
    #             break

    #     if args.past_index and hasattr(self, "_past"):
    #         # Clean the state at the end of training
    #         delattr(self, "_past")

    #     logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
    #     if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
    #         # Wait for everyone to get here so we are sur the model has been saved by process 0.
    #         if is_torch_tpu_available():
    #             xm.rendezvous("load_best_model_at_end")
    #         if args.parallel_mode == ParallelMode.DISTRIBUTED:
    #             dist.barrier()
    #         elif is_sagemaker_mp_enabled():
    #             smp.barrier()

    #         self._load_best_model()

    #     # add remaining tr_loss
    #     self._total_loss_scalar += tr_loss.item()
    #     train_loss = self._total_loss_scalar / self.state.global_step

    #     metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
    #     self.store_flos()
    #     metrics["total_flos"] = self.state.total_flos
    #     metrics["train_loss"] = train_loss

    #     self.is_in_train = False

    #     self._memory_tracker.stop_and_update_metrics(metrics)

    #     self.log(metrics)

    #     run_dir = self._get_output_dir(trial)
    #     checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

    #     # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
    #     if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
    #         for checkpoint in checkpoints_sorted:
    #             if checkpoint != self.state.best_model_checkpoint:
    #                 logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
    #                 shutil.rmtree(checkpoint)

    #     self.control = self.callback_handler.on_train_end(args, self.state, self.control)

    #     return TrainOutput(self.state.global_step, train_loss, metrics)

    # def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
    #     if self.control.should_log:
    #         # MODIFIED: removed --> faster
    #         # if is_torch_tpu_available():
    #         #     xm.mark_step()

    #         logs: Dict[str, float] = {}

    #         # all_gather + mean() to get average loss over all processes
    #         # tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
    #         # MODIFIED: no gather since we train independently
    #         tr_loss_scalar = tr_loss.item()

    #         # reset tr_loss to zero
    #         tr_loss -= tr_loss

    #         loss = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
    #         logs[f"{self.logging_prefix}loss"] = loss
    #         logs["loss"] = loss

    #         logs[f"{self.logging_prefix}learning_rate"] = self._get_learning_rate()
    #         # prepend logging_prefix to epoch
    #         if self.state.epoch is not None:
    #             logs[f"{self.logging_prefix}epoch"] = round(self.state.epoch, 2)
    #         logs[f"{self.logging_prefix}global_step"] = self.state.global_step

    #         self._total_loss_scalar += tr_loss_scalar
    #         self._globalstep_last_logged = self.state.global_step
    #         self.store_flos()

    #         self.log(logs)

    #     metrics = None
    #     if self.control.should_evaluate:
    #         metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
    #         self._report_to_hp_search(trial, self.state.global_step, metrics)

    #     if self.control.should_save:
    #         self._save_checkpoint(model, trial, metrics=metrics)
    #         self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    # def log(self, logs: Dict[str, float]) -> None:
    #     """
    #     Log `logs` on the various objects watching training.

    #     Subclass and override this method to inject custom behavior.

    #     Args:
    #         logs (`Dict[str, float]`):
    #             The values to log.
    #     """
    #     if self.state.epoch is not None:
    #         logs["epoch"] = round(self.state.epoch, 2)

    #     output = {**logs, **{"step": self.state.global_step}}
    #     self.state.log_history.append(output)
    #     # MODIFIED: also log current device
    #     logger.warning(f"{xm.get_ordinal()}: {output}")
    #     self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args
        
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        
        if not self.skip_eval_loss:
            # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
            # while ``train`` is running, cast it to the right dtype first and then put on device
            if not self.is_in_train:
                if args.fp16_full_eval:
                    model = model.to(dtype=torch.float16, device=args.device)
                elif args.bf16_full_eval:
                    model = model.to(dtype=torch.bfloat16, device=args.device)

            batch_size = self.args.eval_batch_size

            logger.warning(f"***** Running {description} *****")
            if has_length(dataloader):
                logger.warning(f"  Num examples = {self.num_examples(dataloader)}")
            else:
                logger.info("  Num examples: Unknown")
            logger.info(f"  Batch size = {batch_size}")

            model.eval()

            self.callback_handler.eval_dataloader = dataloader
            # Do this before wrapping.
            eval_dataset = getattr(dataloader, "dataset", None)

            # MODIFIED: not necessary.
            if is_torch_tpu_available():
                dataloader = pl.MpDeviceLoader(dataloader, args.device)  # .per_device_loader(args.device)

            if args.past_index >= 0:
                self._past = None

            # Initialize containers
            # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
            losses_host = None
            preds_host = None
            labels_host = None
            inputs_host = None

            # losses/preds/labels on CPU (final containers)
            all_losses = None
            all_preds = None
            all_labels = None
            all_inputs = None
            # Will be useful when we have an iterable dataset so don't know its length.

            observed_num_examples = 0
            # Main evaluation loop
            for step, inputs in enumerate(dataloader):
                # Update the observed num examples
                observed_batch_size = find_batch_size(inputs)
                if observed_batch_size is not None:
                    observed_num_examples += observed_batch_size
                    # For batch samplers, batch_size is not known by the dataloader in advance.
                    if batch_size is None:
                        batch_size = observed_batch_size

                # Prediction step
                loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
                inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None

                # MODIFIED: not necessary.
                if is_torch_tpu_available():
                    xm.mark_step()

                # Update containers on host
                if loss is not None:
                    # MODIFIED: do not gather across devices. (different loss on each device)
                    losses = loss.repeat(batch_size)
                    losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
                if labels is not None:
                    labels = self._pad_across_processes(labels)
                    # MODIFIED: do not gather across devices.
                    labels = self._nested_gather(labels)
                    labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
                if inputs_decode is not None:
                    inputs_decode = self._pad_across_processes(inputs_decode)
                    # MODIFIED: do not gather across devices.
                    inputs_decode = self._nested_gather(inputs_decode)
                    inputs_host = (
                        inputs_decode
                        if inputs_host is None
                        else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                    )
                if logits is not None:
                    logits = self._pad_across_processes(logits)
                    # logits = self._nested_gather(logits)
                    if self.preprocess_logits_for_metrics is not None:
                        logits = self.preprocess_logits_for_metrics(logits, labels)
                    preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
                self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

                # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
                if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                    if losses_host is not None:
                        losses = nested_numpify(losses_host)
                        all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                    if preds_host is not None:
                        logits = nested_numpify(preds_host)
                        all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                    if inputs_host is not None:
                        inputs_decode = nested_numpify(inputs_host)
                        all_inputs = (
                            inputs_decode
                            if all_inputs is None
                            else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                        )
                    if labels_host is not None:
                        labels = nested_numpify(labels_host)
                        all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

                    # Set back to None to begin a new accumulation
                    losses_host, preds_host, inputs_host, labels_host = (
                        None,
                        None,
                        None,
                        None,
                    )

            if args.past_index and hasattr(self, "_past"):
                # Clean the state at the end of the evaluation loop
                delattr(self, "_past")

            # Gather all remaining tensors and put them back on the CPU
            if losses_host is not None:
                losses = nested_numpify(losses_host)
                all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
            if preds_host is not None:
                logits = nested_numpify(preds_host)
                all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
            if inputs_host is not None:
                inputs_decode = nested_numpify(inputs_host)
                all_inputs = (
                    inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                )
            if labels_host is not None:
                labels = nested_numpify(labels_host)
                all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

            # Number of samples
            if has_length(eval_dataset):
                num_samples = len(eval_dataset)
            # The instance check is weird and does not actually check for the type, but whether the dataset has the right
            # methods. Therefore we need to make sure it also has the attribute.
            elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
                num_samples = eval_dataset.num_examples
            else:
                if has_length(dataloader):
                    num_samples = self.num_examples(dataloader)
                else:  # both len(dataloader.dataset) and len(dataloader) fail
                    num_samples = observed_num_examples

            # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
            # samplers has been rounded to a multiple of batch_size, so we truncate.
            if all_losses is not None:
                all_losses = all_losses[:num_samples]
            if all_preds is not None:
                all_preds = nested_truncate(all_preds, num_samples)
            if all_labels is not None:
                all_labels = nested_truncate(all_labels, num_samples)
            if all_inputs is not None:
                all_inputs = nested_truncate(all_inputs, num_samples)
        else:
            xm.rendezvous("eval_metrics")
            all_losses, all_preds, all_labels, all_inputs, num_samples = None, None, None, None, 0

        # Metrics!
        # MODIFIED: removed since done in compute_metrics
        xm.rendezvous("eval_metrics")
        # MODIFIED: always compute metrics
        if self.compute_metrics is not None:
            metrics = self.compute_metrics(self)
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            # MODIFIED: no gather, add prefix
            loss = all_losses.mean().item()
            metrics[f"{metric_key_prefix}_{self.logging_prefix}loss"] = loss
            metrics[f"{metric_key_prefix}_loss"] = loss

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(
            predictions=all_preds,
            label_ids=all_labels,
            metrics=metrics,
            num_samples=num_samples,
        )

    def _save_tpu(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        logger.info(f"Saving model checkpoint to {output_dir}")

        # MODIFIED: also save on other devices
        if xm.is_master_ordinal():
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        # # Save a trained model and configuration using `save_pretrained()`.
        # # They can then be reloaded using `from_pretrained()`
        xm.rendezvous("saving_checkpoint")
        if isinstance(self.model, Model):
            actual_model = self.model.backbone
        else:
            actual_model = self.model
        if not isinstance(actual_model, PreTrainedModel):
            if isinstance(unwrap_model(actual_model), PreTrainedModel):
                unwrap_model(actual_model).save_pretrained(
                    output_dir,
                    is_main_process=self.args.should_save,
                    state_dict=actual_model.state_dict(),
                    save_function=xm.save,
                )
            else:
                logger.warning("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                state_dict = actual_model.state_dict()
                xm.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            actual_model.save_pretrained(output_dir, is_main_process=self.args.should_save, save_function=xm.save)
        if self.tokenizer is not None and self.args.should_save:
            self.tokenizer.save_pretrained(output_dir)


class AdapterTrainerCallback(TrainerCallback):
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs.pop("model")
        model_frozen = getattr(model.backbone.base_model, "model_frozen", False)
        if not model_frozen:
            raise ValueError(
                "The pre-trained model weights are not frozen. For training adapters, please call the train_adapter()"
                " method"
            )

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # apply adapter fusion weight regularization on the value matrix
        model = kwargs.pop("model")
        if self.trainer.train_adapter_fusion:
            fusion_reg_loss = model.backbone.base_model.get_fusion_regularization_loss()
            if fusion_reg_loss is not None:
                fusion_reg_loss.backward()
