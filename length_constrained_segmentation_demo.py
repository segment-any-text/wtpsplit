#!/usr/bin/env python3
"""
Length-Constrained Segmentation Demo

Interactive demo for experimenting with length-constrained segmentation.
For detailed documentation, see docs/LENGTH_CONSTRAINTS.md

Usage:
    python length_constrained_segmentation_demo.py              # Run all examples
    python length_constrained_segmentation_demo.py --interactive  # Interactive mode
    python length_constrained_segmentation_demo.py --example news  # Specific example
"""

import argparse
import sys

# =============================================================================
# SETUP
# =============================================================================


def load_model():
    """Load SaT model."""
    from wtpsplit import SaT

    print("Loading model...", end=" ", flush=True)
    sat = SaT("sat-3l-sm", ort_providers=["CPUExecutionProvider"])
    print("✓")
    return sat


# =============================================================================
# DISPLAY UTILITIES
# =============================================================================


class C:
    """ANSI colors."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"


def show_segments(segments, max_length=None, label=""):
    """Display segments with length info and quality indicators."""
    if label:
        print(f"\n{C.BOLD}{label}{C.RESET}")

    for i, seg in enumerate(segments, 1):
        length = len(seg)

        # Check for word cuts (ends with letter preceded by letter)
        has_cut = len(seg) > 1 and seg[-1].isalpha() and seg[-2].isalpha()

        # Status indicator
        if has_cut:
            status = f"{C.YELLOW}~{C.RESET}"  # Word cut
        elif max_length and length > max_length:
            status = f"{C.RED}!{C.RESET}"  # Exceeds max
        else:
            status = f"{C.GREEN}✓{C.RESET}"  # OK

        # Length display
        if max_length:
            len_str = f"[{length:3d}/{max_length}]"
        else:
            len_str = f"[{length:3d}]"

        # Truncate for display
        display = repr(seg[:50]) + ("..." if len(seg) > 50 else "")

        print(f"  {status} {len_str} {display}")


def verify_preservation(original, segments):
    """Check if text is preserved."""
    rejoined = "".join(segments)
    if rejoined == original:
        print(f"  {C.GREEN}✓ Text preserved{C.RESET}")
        return True
    else:
        print(f"  {C.RED}✗ Text NOT preserved!{C.RESET}")
        print(f"    Original: {len(original)} chars")
        print(f"    Rejoined: {len(rejoined)} chars")
        return False


# =============================================================================
# EXAMPLES
# =============================================================================

EXAMPLES = {
    "basic": {
        "name": "Basic Sentences",
        "text": "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump!",
        "configs": [
            {"max_length": 50},
            {"max_length": 80},
            {"min_length": 30, "max_length": 100},
        ],
    },
    "news": {
        "name": "News Article",
        "text": """Breaking News: Scientists at CERN have announced a groundbreaking discovery that could revolutionize our understanding of particle physics. The team, led by Dr. Elena Rodriguez, observed unexpected behavior in proton collisions at energies never before achieved. "This is the most significant finding in our field since the Higgs boson," Dr. Rodriguez stated at a press conference in Geneva. The discovery has implications for theories of dark matter and could lead to new technologies within the next decade.""",
        "configs": [
            {"max_length": 100},
            {"max_length": 150},
            {"max_length": 200},
        ],
    },
    "legal": {
        "name": "Legal Text (Long Sentences)",
        "text": """WHEREAS the Party of the First Part (hereinafter referred to as "Licensor") is the owner of certain intellectual property rights including but not limited to patents, trademarks, copyrights, and trade secrets relating to the technology described herein, and WHEREAS the Party of the Second Part (hereinafter referred to as "Licensee") desires to obtain a license to use, manufacture, and distribute products incorporating said technology, NOW THEREFORE in consideration of the mutual covenants and agreements set forth herein, the parties agree as follows.""",
        "configs": [
            {"max_length": 100},
            {"max_length": 150},
            {"max_length": 250},
        ],
    },
    "technical": {
        "name": "Technical Documentation",
        "text": """The function accepts three parameters: input_data (required), config (optional), and callback (optional). When input_data is a string, it will be parsed as JSON; when it's an object, it will be used directly. The config parameter supports the following options: timeout (default: 30000ms), retries (default: 3), and verbose (default: false). If callback is provided, the function operates asynchronously.""",
        "configs": [
            {"max_length": 80},
            {"max_length": 120},
            {"min_length": 50, "max_length": 150},
        ],
    },
    "stream": {
        "name": "Stream of Consciousness (No Punctuation)",
        "text": """I was walking down the street thinking about what to have for dinner maybe pasta or perhaps something lighter like a salad but then again it was cold outside and soup sounded really good especially that tomato soup from the place around the corner which reminded me I needed to call my mother""",
        "configs": [
            {"max_length": 60},
            {"max_length": 100},
            {"max_length": 150},
        ],
    },
    "dialogue": {
        "name": "Dialogue with Quotes",
        "text": '''"Have you seen the news?" asked Maria. "About the merger?" replied John. "No, I mean about the earthquake." Maria shook her head. "It's terrible." John sighed. "Sometimes I wonder if things will ever get better."''',
        "configs": [
            {"max_length": 50},
            {"max_length": 80},
            {"max_length": 120},
        ],
    },
    "mixed": {
        "name": "Mixed Content (Numbers, Abbreviations)",
        "text": """Dr. Smith earned $125,000 in Q4 2023, a 15.7% increase. The U.S. Department of Labor reported unemployment at 3.5%. Mr. Johnson's company, ABC Corp., plans to expand to the U.K. and E.U. by mid-2024. The CEO stated: "We expect revenues of $50M-$75M." The S&P 500 closed at 4,769.83 pts.""",
        "configs": [
            {"max_length": 80},
            {"max_length": 120},
            {"min_length": 40, "max_length": 160},
        ],
    },
    "priors": {
        "name": "Prior Functions Comparison",
        "text": "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten. Eleven. Twelve.",
        "configs": [
            {"max_length": 100, "prior_type": "uniform"},
            {"max_length": 100, "prior_type": "gaussian", "prior_kwargs": {"target_length": 30, "spread": 10}},
            {"max_length": 100, "prior_type": "gaussian", "prior_kwargs": {"target_length": 60, "spread": 15}},
        ],
    },
    "algorithms": {
        "name": "Algorithm Comparison (Viterbi vs Greedy)",
        "text": "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump! The five boxing wizards jump quickly. Sphinx of black quartz, judge my vow.",
        "configs": [
            {"max_length": 80, "algorithm": "viterbi"},
            {"max_length": 80, "algorithm": "greedy"},
        ],
    },
}


def run_example(sat, name, example):
    """Run a single example."""
    print(f"\n{'=' * 70}")
    print(f"{C.BOLD}{example['name']}{C.RESET}")
    print(f"{'=' * 70}")
    print(f"\n{C.GRAY}Text ({len(example['text'])} chars):{C.RESET}")
    print(f'  "{example["text"][:80]}{"..." if len(example["text"]) > 80 else ""}"')

    for config in example["configs"]:
        # Build label
        parts = []
        if "max_length" in config:
            parts.append(f"max={config['max_length']}")
        if "min_length" in config:
            parts.append(f"min={config['min_length']}")
        if config.get("algorithm"):
            parts.append(f"algo={config['algorithm']}")
        if config.get("prior_type") and config["prior_type"] != "uniform":
            parts.append(f"prior={config['prior_type']}")
        label = ", ".join(parts)

        # Run segmentation
        segments = sat.split(example["text"], threshold=0.025, **config)

        show_segments(segments, config.get("max_length"), label)
        verify_preservation(example["text"], segments)


def run_all_examples(sat):
    """Run all examples."""
    print(f"\n{C.CYAN}{'=' * 70}")
    print("  LENGTH-CONSTRAINED SEGMENTATION EXAMPLES")
    print(f"{'=' * 70}{C.RESET}")

    for name, example in EXAMPLES.items():
        run_example(sat, name, example)

    print(f"\n{C.CYAN}{'=' * 70}{C.RESET}")
    print(f"\nFor interactive mode: {C.BOLD}python {sys.argv[0]} --interactive{C.RESET}")
    print(f"For documentation: {C.BOLD}see docs/LENGTH_CONSTRAINTS.md{C.RESET}")


# =============================================================================
# INTERACTIVE MODE
# =============================================================================


def interactive_mode(sat):
    """Interactive segmentation playground."""
    print(f"\n{C.CYAN}{'=' * 70}")
    print("  INTERACTIVE MODE")
    print(f"{'=' * 70}{C.RESET}")

    print("""
Commands:
  <text>     - Segment the text
  max=N      - Set max_length
  min=N      - Set min_length
  algo=X     - Set algorithm (viterbi/greedy)
  prior=X    - Set prior (uniform/gaussian/polynomial)
  reset      - Reset to defaults
  examples   - List available examples
  run NAME   - Run an example
  q          - Quit
""")

    # Settings
    settings = {
        "max_length": 100,
        "min_length": 1,
        "algorithm": "viterbi",
        "prior_type": "uniform",
        "prior_kwargs": None,
    }

    while True:
        # Show current settings
        print(
            f"\n{C.GRAY}[max={settings['max_length']}, min={settings['min_length']}, "
            f"algo={settings['algorithm']}, prior={settings['prior_type']}]{C.RESET}"
        )

        try:
            user_input = input(f"{C.BOLD}> {C.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Commands
        if user_input.lower() == "q":
            print("Goodbye!")
            break

        if user_input.lower() == "reset":
            settings = {
                "max_length": 100,
                "min_length": 1,
                "algorithm": "viterbi",
                "prior_type": "uniform",
                "prior_kwargs": None,
            }
            print("Settings reset.")
            continue

        if user_input.lower() == "examples":
            print("\nAvailable examples:")
            for name, ex in EXAMPLES.items():
                print(f"  {name:12s} - {ex['name']}")
            continue

        if user_input.lower().startswith("run "):
            name = user_input[4:].strip()
            if name in EXAMPLES:
                run_example(sat, name, EXAMPLES[name])
            else:
                print(f"Unknown example: {name}")
            continue

        if user_input.startswith("max="):
            try:
                settings["max_length"] = int(user_input[4:])
                print(f"max_length = {settings['max_length']}")
            except ValueError:
                print("Invalid number")
            continue

        if user_input.startswith("min="):
            try:
                settings["min_length"] = int(user_input[4:])
                print(f"min_length = {settings['min_length']}")
            except ValueError:
                print("Invalid number")
            continue

        if user_input.startswith("algo="):
            algo = user_input[5:].strip()
            if algo in ["viterbi", "greedy"]:
                settings["algorithm"] = algo
                print(f"algorithm = {algo}")
            else:
                print("Unknown algorithm (use: viterbi, greedy)")
            continue

        if user_input.startswith("prior="):
            prior = user_input[6:].strip()
            if prior in ["uniform", "gaussian", "polynomial"]:
                settings["prior_type"] = prior if prior != "polynomial" else "clipped_polynomial"
                if prior == "gaussian":
                    settings["prior_kwargs"] = {"target_length": settings["max_length"] * 0.7, "spread": 15}
                elif prior == "polynomial":
                    settings["prior_kwargs"] = {"target_length": settings["max_length"] * 0.7, "spread": 30}
                else:
                    settings["prior_kwargs"] = None
                print(f"prior = {prior}")
            else:
                print("Unknown prior (use: uniform, gaussian, polynomial)")
            continue

        # Treat as text to segment
        text = user_input

        try:
            kwargs = {
                "threshold": 0.025,
                "max_length": settings["max_length"],
                "min_length": settings["min_length"],
                "algorithm": settings["algorithm"],
                "prior_type": settings["prior_type"],
            }
            if settings["prior_kwargs"]:
                kwargs["prior_kwargs"] = settings["prior_kwargs"]

            segments = sat.split(text, **kwargs)

            print(f"\n{C.BOLD}Result: {len(segments)} segments{C.RESET}")
            show_segments(segments, settings["max_length"])
            verify_preservation(text, segments)

        except Exception as e:
            print(f"{C.RED}Error: {e}{C.RESET}")


# =============================================================================
# PROBABILITY VISUALIZATION
# =============================================================================


def show_probabilities(sat):
    """Visualize model probabilities for a sample text."""
    print(f"\n{C.CYAN}{'=' * 70}")
    print("  PROBABILITY VISUALIZATION")
    print(f"{'=' * 70}{C.RESET}")

    text = "The quick brown fox jumps. Pack my box with jugs. How vexingly quick!"

    print(f'\n{C.BOLD}Text:{C.RESET} "{text}"')

    # Get probabilities
    probs = list(sat.predict_proba([text]))[0]

    # Build visualization
    viz = ""
    for p in probs:
        if p > 0.9:
            viz += f"{C.GREEN}█{C.RESET}"
        elif p > 0.5:
            viz += f"{C.YELLOW}▓{C.RESET}"
        elif p > 0.1:
            viz += f"{C.GRAY}▒{C.RESET}"
        else:
            viz += f"{C.GRAY}░{C.RESET}"

    print(f"\n{C.BOLD}Probabilities:{C.RESET}")
    print(f"  {text}")
    print(f"  {viz}")
    print(
        f"\n  Legend: {C.GREEN}█{C.RESET}>0.9  {C.YELLOW}▓{C.RESET}>0.5  {C.GRAY}▒{C.RESET}>0.1  {C.GRAY}░{C.RESET}≤0.1"
    )

    # Show high-probability positions
    print(f"\n{C.BOLD}Detected boundaries (prob > 0.5):{C.RESET}")
    for i, p in enumerate(probs):
        if p > 0.5:
            ctx = text[max(0, i - 5) : i + 3]
            print(f'  Position {i:2d}: p={p:.3f}  "...{ctx}..."')


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Length-Constrained Segmentation Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python %(prog)s                    # Run all examples
  python %(prog)s --interactive      # Interactive playground
  python %(prog)s --example news     # Run specific example
  python %(prog)s --probs            # Show probability visualization

Available examples: """
        + ", ".join(EXAMPLES.keys()),
    )
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("-e", "--example", choices=list(EXAMPLES.keys()), help="Run specific example")
    parser.add_argument("-p", "--probs", action="store_true", help="Show probability visualization")

    args = parser.parse_args()

    sat = load_model()

    if args.interactive:
        interactive_mode(sat)
    elif args.example:
        run_example(sat, args.example, EXAMPLES[args.example])
    elif args.probs:
        show_probabilities(sat)
    else:
        run_all_examples(sat)


if __name__ == "__main__":
    main()
