"""Detector training pipeline: dataset fetchers, per-dataset converters and a builder that
merges the team's drone datasets into one YOLO dataset for YOLO26 training.

    python tools/fetch_datasets.py --list        # what is present, what still needs downloading
    python tools/build_dataset.py                # raw datasets -> datasets/aimbot_drones (YOLO format)
    python tools/train_yolo26.py                 # YOLO26 training + per-dataset validation

See docs/TRAINING.md.
"""
