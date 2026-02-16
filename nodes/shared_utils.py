"""Shared utility functions for MotionCapture nodes."""

import os

import folder_paths


def next_sequential_filename(directory, prefix, ext):
    """Find the next sequential filename like prefix_0001.ext, prefix_0002.ext, etc."""
    existing = sorted(directory.glob(f"{prefix}_*{ext}"))
    max_num = 0
    for f in existing:
        stem = f.stem
        suffix = stem[len(prefix) + 1:]
        try:
            max_num = max(max_num, int(suffix))
        except ValueError:
            pass
    return f"{prefix}_{max_num + 1:04d}{ext}"


def resolve_file_path(file_path):
    """Resolve a file path that may have [output] prefix to an absolute path.

    Checks output folder (if prefixed with [output]), input folder, and absolute paths.
    Returns None if the file is not found.
    """
    if file_path.startswith("[output] "):
        clean_path = file_path.replace("[output] ", "")
        output_dir = folder_paths.get_output_directory()
        output_path = os.path.join(output_dir, clean_path)
        if os.path.exists(output_path):
            return output_path
    else:
        input_dir = folder_paths.get_input_directory()
        input_path = os.path.join(input_dir, file_path)
        if os.path.exists(input_path):
            return input_path

    if os.path.exists(file_path):
        return file_path

    return None
