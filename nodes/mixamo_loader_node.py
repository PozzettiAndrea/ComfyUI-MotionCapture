"""
LoadMixamoCharacter Node - Load Mixamo-rigged FBX characters.

Searches both input and output folders for .fbx files.
"""

import os
import folder_paths


class LoadMixamoCharacter:
    """
    Load a Mixamo-rigged FBX character.

    Searches both input and output folders for .fbx files.
    Returns the resolved file path.
    """

    @classmethod
    def INPUT_TYPES(cls):
        fbx_files = cls.get_fbx_files()
        if not fbx_files:
            fbx_files = ["No .fbx files found"]
        return {
            "required": {
                "fbx_file": (fbx_files, {
                    "tooltip": "FBX file containing Mixamo-rigged character"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("fbx_path", "info")
    FUNCTION = "load_mixamo"
    CATEGORY = "MotionCapture/Mixamo"

    @classmethod
    def _scan_fbx(cls, base_dir, prefix=""):
        """Recursively scan directory for .fbx files."""
        fbx_files = []
        if not os.path.exists(base_dir):
            return fbx_files
        for root, _dirs, files in os.walk(base_dir):
            for file in sorted(files):
                if file.lower().endswith('.fbx'):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, base_dir)
                    if prefix:
                        fbx_files.append(f"{prefix}{rel_path}")
                    else:
                        fbx_files.append(rel_path)
        return fbx_files

    @classmethod
    def get_fbx_files(cls):
        """Get list of .fbx files in input and output folders."""
        fbx_files = []

        # Scan input folder
        input_dir = folder_paths.get_input_directory()
        fbx_files.extend(cls._scan_fbx(input_dir))

        # Scan output folder
        output_dir = folder_paths.get_output_directory()
        fbx_files.extend(cls._scan_fbx(output_dir, prefix="[output] "))

        return fbx_files

    @classmethod
    def IS_CHANGED(cls, fbx_file):
        full_path = cls._resolve_file_path(fbx_file)
        if full_path and os.path.exists(full_path):
            return os.path.getmtime(full_path)
        return fbx_file

    @classmethod
    def _resolve_file_path(cls, fbx_file):
        if fbx_file.startswith("[output] "):
            clean_path = fbx_file.replace("[output] ", "")
            output_dir = folder_paths.get_output_directory()
            output_path = os.path.join(output_dir, clean_path)
            if os.path.exists(output_path):
                return output_path
        else:
            input_dir = folder_paths.get_input_directory()
            input_path = os.path.join(input_dir, fbx_file)
            if os.path.exists(input_path):
                return input_path

        if os.path.exists(fbx_file):
            return fbx_file

        return None

    def load_mixamo(self, fbx_file):
        full_path = self._resolve_file_path(fbx_file)
        if full_path is None:
            raise FileNotFoundError(f"Mixamo FBX file not found: {fbx_file}")

        file_size = os.path.getsize(full_path) / (1024 * 1024)  # MB
        source = "output" if fbx_file.startswith("[output] ") else "input"

        info = (
            f"Mixamo Character Loaded\n"
            f"File: {fbx_file}\n"
            f"Source: {source}\n"
            f"Full path: {full_path}\n"
            f"Size: {file_size:.2f} MB\n"
        )

        print(f"[LoadMixamoCharacter] Selected: {full_path}")
        return (full_path, info)


NODE_CLASS_MAPPINGS = {
    "LoadMixamoCharacter": LoadMixamoCharacter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadMixamoCharacter": "Load Mixamo Character",
}
