"""
Shared multimodal feature extraction used across the stack.

This package sits alongside ``webapp/`` (ingestion + API) and ``notebooks/`` (EDA
and demos): speech (librosa + Praat), handwriting images (OpenCV / skimage),
and gait video (OpenCV), matching the ingestion → feature extraction stage of
the system architecture.
"""
