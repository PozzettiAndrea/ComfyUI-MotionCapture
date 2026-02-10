"""
LoadSMPL Node - Load SMPL motion data from disk.

Loads .npz files containing SMPL motion parameters from ComfyUI folders.
"""

import os
import folder_paths


class LoadSMPL:
    """
    Select an SMPL motion .npz file.

    Automatically searches both input and output folders.
    Returns the resolved file path.
    """

    SUPPORTED_EXTENSIONS = ['.npz']

    @classmethod
    def INPUT_TYPES(cls):
        npz_files = cls.get_npz_files()
        if not npz_files:
            npz_files = ["No NPZ files found"]
        return {
            "required": {
                "file_path": (npz_files, {
                    "tooltip": "NPZ file containing SMPL motion parameters. Files prefixed with [output] are from the output folder."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "load_smpl"
    CATEGORY = "MotionCapture/SMPL"

    @classmethod
    def get_npz_files(cls):
        """Get list of available NPZ files in input and output folders."""
        npz_files = []

        # Scan input folder
        input_dir = folder_paths.get_input_directory()
        if os.path.exists(input_dir):
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in cls.SUPPORTED_EXTENSIONS):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, input_dir)
                        npz_files.append(rel_path)

        # Scan output folder
        output_dir = folder_paths.get_output_directory()
        if os.path.exists(output_dir):
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in cls.SUPPORTED_EXTENSIONS):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, output_dir)
                        npz_files.append(f"[output] {rel_path}")

        return sorted(npz_files)

    @classmethod
    def IS_CHANGED(cls, file_path):
        """Force re-execution when file changes."""
        full_path = cls._resolve_file_path(file_path)
        if full_path and os.path.exists(full_path):
            return os.path.getmtime(full_path)
        return file_path

    @classmethod
    def _resolve_file_path(cls, file_path):
        """Resolve the full path to the NPZ file."""
        # Output files are prefixed with [output]
        if file_path.startswith("[output] "):
            clean_path = file_path.replace("[output] ", "")
            output_dir = folder_paths.get_output_directory()
            output_path = os.path.join(output_dir, clean_path)
            if os.path.exists(output_path):
                return output_path
        else:
            # Input files
            input_dir = folder_paths.get_input_directory()
            input_path = os.path.join(input_dir, file_path)
            if os.path.exists(input_path):
                return input_path

        # Try as absolute path
        if os.path.exists(file_path):
            return file_path

        return None

    def load_smpl(self, file_path):
        """
        Resolve and return the full path to the NPZ file.

        Args:
            file_path: Path to NPZ file (may include [output] prefix)

        Returns:
            tuple: (full_path,)
        """
        full_path = self._resolve_file_path(file_path)

        if full_path is None:
            raise FileNotFoundError(
                f"SMPL file not found: {file_path}\n"
                f"Searched in both input and output folders.\n"
                f"Make sure the file exists."
            )

        print(f"[LoadSMPL] Selected: {full_path}")
        return (full_path,)


NODE_CLASS_MAPPINGS = {
    "LoadSMPL": LoadSMPL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSMPL": "Load SMPL Motion",
}
