"""
cli.py — Command-line interface for the Rosa-sinensis leaf health detection system.

Usage:
  # Two-image analysis (recommended, per capture protocol):
  python cli.py --backlit path/to/backlit.jpg --frontlit path/to/frontlit.jpg

  # Single-image analysis (for testing only):
  python cli.py --image path/to/leaf.jpg

  # With output directory:
  python cli.py --backlit back.jpg --frontlit front.jpg --output results/

  # With custom image ID:
  python cli.py --image leaf.jpg --id leaf_042

This is the primary user-facing entry point for the system.
"""

import argparse
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pipeline import run_pipeline, run_pipeline_single_image


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rosa-sinensis Leaf Health Detection System\n"
            "Rule-based analysis of leaf images to detect nutrient deficiency patterns."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python cli.py --backlit leaf_001_backlit.jpg --frontlit leaf_001_frontlit.jpg\n"
            "  python cli.py --image leaf_sample.jpg --output results/\n"
            "  python cli.py --backlit back.jpg --frontlit front.jpg --id leaf_042 --output out/\n"
        ),
    )

    # Image input (mutually exclusive: two-image or single-image mode)
    input_group = parser.add_argument_group('Image Input')
    input_group.add_argument(
        '--backlit', type=str, default=None,
        help='Path to backlit (transmitted light) leaf image. Use with --frontlit.'
    )
    input_group.add_argument(
        '--frontlit', type=str, default=None,
        help='Path to front-lit (daylight) leaf image. Use with --backlit.'
    )
    input_group.add_argument(
        '--image', type=str, default=None,
        help='Path to a single leaf image (used for both analyses — testing only).'
    )

    # Options
    parser.add_argument(
        '--id', type=str, default=None,
        help='Custom image ID (defaults to filename stem).'
    )
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='Output directory for reports and debug images.'
    )
    parser.add_argument(
        '--no-debug', action='store_true',
        help='Skip saving debug overlay and annotated images.'
    )
    parser.add_argument(
        '--select-roi', action='store_true',
        help='Interactively draw a circle to select the region of interest.'
    )

    args = parser.parse_args()

    # ── Validate arguments ─────────────────────────────────────────────
    has_paired = args.backlit is not None or args.frontlit is not None
    has_single = args.image is not None

    if has_paired and has_single:
        parser.error("Cannot use --image with --backlit/--frontlit. Choose one mode.")

    if has_paired:
        if args.backlit is None or args.frontlit is None:
            parser.error("--backlit and --frontlit must both be specified for paired mode.")
        if not os.path.isfile(args.backlit):
            parser.error(f"Backlit image not found: {args.backlit}")
        if not os.path.isfile(args.frontlit):
            parser.error(f"Frontlit image not found: {args.frontlit}")
    elif has_single:
        if not os.path.isfile(args.image):
            parser.error(f"Image not found: {args.image}")
    else:
        parser.error("Provide either --backlit + --frontlit, or --image.")

    # ── Run pipeline ───────────────────────────────────────────────────
    print("")
    print("==================================================================")
    print("|  Rosa-sinensis Leaf Health Detection System                    |")
    print("|  Rule-based analysis * No ML * Interpretable output            |")
    print("==================================================================")
    print("")

    try:
        if has_paired:
            result = run_pipeline(
                backlit_path=args.backlit,
                frontlit_path=args.frontlit,
                image_id=args.id,
                output_dir=args.output,
                save_debug=not args.no_debug,
                use_interactive_roi=args.select_roi,
            )
        else:
            result = run_pipeline_single_image(
                image_path=args.image,
                image_id=args.id,
                output_dir=args.output,
                save_debug=not args.no_debug,
                use_interactive_roi=args.select_roi,
            )

        # Print the full text report
        print("")
        print(result['report_text'])

        # Summary of saved files
        if result['output_files']:
            print("")
            print(f"Saved {len(result['output_files'])} output file(s):")
            for path in result['output_files']:
                print(f"  -> {path}")

    except FileNotFoundError as e:
        print(f"\nX Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\nX Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nX Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
