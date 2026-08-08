from scripts.prepare_stage2_runs import render_pair


def test_render_pair_changes_only_replay_treatment():
    base = {
        "data_path": "old.pth",
        "model_name_or_path": "old",
        "tokenizer_name_or_path": "old",
        "output_dir": "old",
        "no_sm_corruption": True,
        "eval_strategy": "no",
    }
    arms = render_pair(
        base,
        model="checkpoint",
        tokenizer="tokenizer",
        data_path="abc.pth",
        replay_data_path="historical.pth",
        replay_fraction=0.5,
        output_root="runs/test",
        seed=13,
        max_steps=300,
    )
    baseline = arms["no_replay"]
    replay = arms["replay"]
    differing = {
        key
        for key in set(baseline) | set(replay)
        if baseline.get(key) != replay.get(key)
    }
    assert differing == {"output_dir", "replay_data_path", "replay_fraction"}
    assert replay["replay_fraction"] == 0.5
    assert baseline["seed"] == replay["seed"] == 13


def test_render_pair_can_select_character_head():
    arms = render_pair(
        {"no_sm_corruption": True},
        model="checkpoint",
        tokenizer="tokenizer",
        data_path="abc.pth",
        replay_data_path="historical.pth",
        replay_fraction=0.5,
        output_root="runs/test",
        seed=13,
        max_steps=300,
        use_character_head=True,
        character_head_init="random",
    )
    for arm in arms.values():
        assert arm["use_character_head"] is True
        assert arm["character_head_init"] == "random"
