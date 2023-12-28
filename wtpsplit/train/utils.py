import logging
import os

logger = logging.getLogger(__name__)


def cleanup_cache_files(datasets) -> int:
    """Clean up all cache files in the dataset cache directory, except those currently used by any of the provided datasets.

    Args:
        datasets (List[Dataset]): A list of dataset objects.

    Be careful when running this command that no other process is currently using other cache files.

    Returns:
        int: Number of removed files.
    """
    if not datasets:
        return 0

    # Collect all current cache files from the provided datasets
    current_cache_files = set()
    for dataset in datasets:
        dataset_cache_files = [os.path.abspath(cache_file["filename"]) for cache_file in dataset.cache_files]
        current_cache_files.update(dataset_cache_files)
    logger.warning(f"Found {len(current_cache_files)} cache files used by the provided datasets.")

    if not current_cache_files:
        return 0

    # Assuming all datasets have cache files in the same directory
    cache_directory = os.path.dirname(next(iter(current_cache_files)))

    files = os.listdir(cache_directory)
    files_to_remove = []
    for f_name in files:
        full_name = os.path.abspath(os.path.join(cache_directory, f_name))
        if f_name.startswith("cache-") and f_name.endswith(".arrow") and full_name not in current_cache_files:
            files_to_remove.append(full_name)

    for file_path in files_to_remove:
        logger.warning(f"Removing {file_path}")
        os.remove(file_path)

    return len(files_to_remove)
