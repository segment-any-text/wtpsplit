import os
from typing import Dict

import numpy as np
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    TRAINING_ARGS_NAME,
    WEIGHTS_NAME,
    DataLoader,
    EvalLoopOutput,
    IterableDatasetShard,
    List,
    Optional,
    deepspeed_init,
    denumpify_detensorize,
    find_batch_size,
    has_length,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    nested_concat,
    nested_numpify,
    nested_truncate,
    get_parameter_names,
)
from transformers.modeling_utils import unwrap_model

from wtpsplit.train.utils import Model

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm  # noqa: F401
    import torch_xla.debug.metrics as met  # noqa: F401
    import torch_xla.distributed.parallel_loader as pl  # noqa: F401

from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from adapters.composition import AdapterCompositionBlock, Fuse

import re
from typing import Callable, Tuple, Union

from torch.utils.data.dataset import Dataset

from transformers import Trainer, __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments
from transformers.utils import CONFIG_NAME, logging


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp


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
        logging_suffix="",
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

        self.logging_suffix = logging_suffix
        
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
            
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

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

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.warning(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

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

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
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

        # Metrics!
        # MODIFIED: always compute metrics
        if self.compute_metrics is not None:
            metrics = self.compute_metrics(self)
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

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
        logger.warning(f"Saving model checkpoint to {output_dir}")

        if xm.is_master_ordinal():
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
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