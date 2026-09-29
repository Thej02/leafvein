"""
severityClass/pipeline.py — High-level pipeline module for running end-to-end diagnosis
with interactive ROI tracing, vein skeleton confirmation, and severity evaluation.
"""

from severityClass.cli import run_severity_pipeline, print_results

__all__ = [
    'run_severity_pipeline',
    'print_results',
]

if __name__ == '__main__':
    from severityClass.cli import main
    main()
