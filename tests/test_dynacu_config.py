from nightwatch.config import load_config, method_config


def test_main_config_is_49_tasks_by_3_with_the_paper_arms():
    cfg = load_config("dynacu")
    assert cfg.tasks.subset == "visual" and cfg.repeats == 3
    assert list(cfg.run_methods) == ["standard", "aoi_visual_no_dom", "pusv", "nightwatch_full"]
    assert method_config(cfg, "aoi_visual_no_dom").get("native_no_dom")


def test_intervention_arms_remove_one_intervention_each():
    cfg = load_config("dynacu_ablations")
    full = method_config(cfg, "nightwatch_full").to_dict()
    assert method_config(cfg, "nightwatch_no_interrupt").to_dict() == dict(full, name="nightwatch_no_interrupt", interrupt=False)
    assert method_config(cfg, "nightwatch_no_void").to_dict() == dict(full, name="nightwatch_no_void", invalidate=False)
    no_wake = method_config(cfg, "nightwatch_no_wake").to_dict()
    assert no_wake == dict(full, name="nightwatch_no_wake", post_cfg=dict(full["post_cfg"], s1_wake=False))


def test_watcher_replacement_covers_both_tiers():
    cfg = load_config("dynacu_ablations")
    full = method_config(cfg, "nightwatch_full").to_dict()
    for name in ["watcher_d3lite_zero", "watcher_jev", "watcher_d3_27b"]:
        arm = method_config(cfg, name).to_dict()
        assert arm["system1"] != full["system1"]
        assert arm["system1_fast"] != full["system1_fast"]
        assert arm == dict(full, name=name, system1=arm["system1"], system1_fast=arm["system1_fast"])


def test_osworld_config_is_100_tasks_by_3():
    import json
    from pathlib import Path
    cfg = load_config("osworld")
    meta = json.loads((Path(__file__).resolve().parents[1] / "configs/tasks/osworld_100.json").read_text())
    assert sum(map(len, meta.values())) == 100 and cfg.repeats == 3
    assert list(cfg.run_methods) == ["fixed_0", "aoi_visual", "pusv", "nightwatch_full"]
