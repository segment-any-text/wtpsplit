"""
Compare captured output JSONs from main+tf4, pr+tf4, pr+tf5.
Supports both legacy format (sat.splits / sat.proba) and case-based format (sat.cases).
Usage: python scripts/compare_outputs.py main_tf4.json pr_tf4.json pr_tf5.json
"""

import json
import sys

RTOL = 1e-4
ATOL = 1e-5


def _compare_float_lists(a, b, path="", rtol=RTOL, atol=ATOL):
    errs = []
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            errs.append(f"{path}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            errs.extend(_compare_float_lists(x, y, f"{path}[{i}]", rtol, atol))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if abs(a - b) > atol + rtol * abs(b):
            errs.append(f"{path}: {a} vs {b}")
    else:
        if type(a) is not type(b) or a != b:
            errs.append(f"{path}: type or value mismatch {type(a).__name__} {a!r} vs {type(b).__name__} {b!r}")
    return errs


def _compare_splits(sa, sb, path):
    """Compare split outputs (list of strings, or list of list of strings for paragraphs/batched)."""
    errs = []
    if isinstance(sa, list) and isinstance(sb, list):
        if len(sa) != len(sb):
            errs.append(f"{path}: length {len(sa)} vs {len(sb)}")
        for i, (xa, xb) in enumerate(zip(sa, sb)):
            if isinstance(xa, list) and isinstance(xb, list):
                errs.extend(_compare_splits(xa, xb, f"{path}[{i}]"))
            elif xa != xb:
                errs.append(f"{path}[{i}]: {xa!r} != {xb!r}")
    else:
        if sa != sb:
            errs.append(f"{path}: {sa!r} != {sb!r}")
    return errs


def _compare_case(ca, cb, case_name):
    errs = []
    keys = set(ca.keys()) | set(cb.keys())
    for k in keys:
        if k not in ca:
            errs.append(f"{case_name}.{k}: missing in first")
            continue
        if k not in cb:
            errs.append(f"{case_name}.{k}: missing in second")
            continue
        va, vb = ca[k], cb[k]
        if k in ("splits", "splits_list") or k.startswith("splits"):
            errs.extend(_compare_splits(va, vb, f"{case_name}.{k}"))
        elif k == "proba":
            errs.extend(_compare_float_lists(va, vb, f"{case_name}.proba"))
        else:
            if va != vb:
                errs.append(f"{case_name}.{k}: {va!r} != {vb!r}")
    return errs


def compare(data_a, data_b):
    issues = []
    for top_key in ("sat", "sat_sm", "wtp", "wtp_canine"):
        if top_key not in data_a and top_key not in data_b:
            continue
        if "error" in data_a.get(top_key, {}) or "error" in data_b.get(top_key, {}):
            ea = data_a.get(top_key, {}).get("error")
            eb = data_b.get(top_key, {}).get("error")
            if ea != eb:
                issues.append(f"{top_key}: error {ea!r} vs {eb!r}")
            continue

        oa = data_a.get(top_key, {}).get("cases", data_a.get(top_key))
        ob = data_b.get(top_key, {}).get("cases", data_b.get(top_key))

        # Legacy format (splits / proba at top level)
        if "cases" not in data_a.get(top_key, {}) and "cases" not in data_b.get(top_key, {}):
            if "splits" in oa or "splits" in ob:
                issues.extend(_compare_splits(oa.get("splits", []), ob.get("splits", []), f"{top_key}.splits"))
            if "proba" in oa or "proba" in ob:
                for i, (pa, pb) in enumerate(zip(oa.get("proba", []), ob.get("proba", []))):
                    issues.extend(_compare_float_lists(pa, pb, f"{top_key}.proba[{i}]"))
            continue

        # Case-based format
        cases = set(oa.keys()) | set(ob.keys())
        cases.discard("model")
        # When adapter not available on one side, one may have error and the other splits (skip as known)
        adapter_optional_cases = {"basic_with_adapter_ud"}
        for case_name in sorted(cases):
            ca = oa.get(case_name, {})
            cb = ob.get(case_name, {})
            if not ca and not cb:
                continue
            has_error_a = "error" in ca and not any(k in ca for k in ("splits", "proba"))
            has_error_b = "error" in cb and not any(k in cb for k in ("splits", "proba"))
            if case_name in adapter_optional_cases and (has_error_a != has_error_b):
                continue  # adapter availability differs, skip this case
            issues.extend(_compare_case(ca, cb, f"{top_key}.{case_name}"))

    return issues


def main():
    if len(sys.argv) < 4:
        print("Usage: compare_outputs.py main_tf4.json pr_tf4.json pr_tf5.json", file=sys.stderr)
        sys.exit(2)
    main_tf4_path, pr_tf4_path, pr_tf5_path = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(main_tf4_path) as f:
        main_tf4 = json.load(f)
    with open(pr_tf4_path) as f:
        pr_tf4 = json.load(f)
    with open(pr_tf5_path) as f:
        pr_tf5 = json.load(f)

    print("Versions:")
    print(f"  main_tf4: transformers={main_tf4.get('transformers')}")
    print(f"  pr_tf4:   transformers={pr_tf4.get('transformers')}")
    print(f"  pr_tf5:   transformers={pr_tf5.get('transformers')}")
    print()

    all_ok = True

    issues_14_24 = compare(main_tf4, pr_tf4)
    if issues_14_24:
        print("main_tf4 vs pr_tf4: DIFFERENCES")
        for i in issues_14_24:
            print(" ", i)
        all_ok = False
    else:
        print("main_tf4 vs pr_tf4: OK (all cases match)")

    issues_24_25 = compare(pr_tf4, pr_tf5)
    if issues_24_25:
        print("pr_tf4 vs pr_tf5: DIFFERENCES")
        for i in issues_24_25:
            print(" ", i)
        all_ok = False
    else:
        print("pr_tf4 vs pr_tf5: OK (all cases match)")

    print()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
