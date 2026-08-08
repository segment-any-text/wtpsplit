"""Arguments shared by Stage 2 and Stage 3 sentence training."""

from dataclasses import dataclass


@dataclass
class SentenceTrainingArguments:
    block_size: int = 256
    num_layers: int = 12
    lim_lookahead: bool = False
    without_pretraining: bool = False
    no_sm_corruption: bool = False
    use_character_head: bool = False
    balance_character_loss: bool = True
    character_head_init: str = "identity"
    run_final_evaluation: bool = False
    evaluation_only: bool = False
    checkpoint_milestones: str | None = None
    model_name_or_path: str | None = None
    tokenizer_name_or_path: str | None = None
    data_path: str = "data/all_data_11_05-all.pth"
    training_dataset: str | None = None
    languages: str | None = None
    max_train_sentences_per_dataset: int | None = None
    max_eval_instances_per_dataset: int = 200
    lookahead: int | None = None

    # Stage 2 retention experiments can draw a fixed fraction of batches from
    # a second sentence corpus. The two corpora are loaded and packed
    # independently, so language identifiers may overlap without one silently
    # replacing the other.
    replay_data_path: str | None = None
    replay_fraction: float = 0.0
    replay_training_dataset: str | None = None
    replay_languages: str | None = None
    max_replay_train_sentences_per_dataset: int | None = None
