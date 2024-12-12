import os
from pathlib import Path
import coolname

def generate_unique_name(experiment_path: Path, slug_length: int = 4) -> str:
    """
    Generate a unique name for an experiment in the given path.

    Args:
        experiment_path (Path): The directory to check for existing names.
        slug_length (int): The number of parts in the generated slug. Default is 4.

    Returns:
        str: A unique name that does not already exist in the directory.
    """
    while True:
        # Generate a new slug
        exp_name = coolname.generate_slug(slug_length)
        # Check if the name already exists in the directory
        if not (experiment_path / exp_name).exists():
            return exp_name