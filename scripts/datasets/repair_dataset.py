#!/usr/bin/env python3

"""
This script will process all parquets files in the specified folder by correcting the episode_index column.
We then compute the statistics using the lerobot_stats_compute.py script.
"""

import os
import re
import sys
import glob
import tyro
import json
import subprocess
import numpy as np
import pandas as pd
from dataclasses import dataclass
from loguru import logger


@dataclass
class Config:
    """
    If no indexes_to_delete is provided, will attempt to repair the dataset.
    If indexes_to_delete is provided, will delete the specified indexes from the dataset and repair the rest.
    """

    dataset_path: str
    """the path to the dataset to repair"""
    indexes_to_delete: str | None = None
    """the indexes to delete, comma separated"""
    videos: bool = False
    """whether to process the videos"""


def check_version(info_path):
    if not os.path.exists(info_path):
        logger.error(f"Error: {info_path} does not exist")
        sys.exit(1)
    with open(info_path, "r") as f:
        try:
            info = json.load(f)
            if "codebase_version" not in info:
                logger.error(f"Error: {info_path} file does not contain a codebase_version key")
                sys.exit(1)
            elif info["codebase_version"] != "v2.0" and info["codebase_version"] != "v2.1":
                logger.error(
                    f"Error: {info_path} is not a v2.0 or v2.1 dataset, found {info['codebase_version']}"
                )
                sys.exit(1)
        except json.JSONDecodeError:
            logger.error(f"Error: {info_path} is not a valid JSON file")
            sys.exit(1)
    return info["codebase_version"]


def delete_DS_Store(dataset_path):
    """
    Delete all .DS_Store files in the given dataset path and its subdirectories.
    """
    logger.info("Deleting .DS_Store files...")
    ds_store_files = glob.glob(
        os.path.join(dataset_path, "**", ".DS_Store"), recursive=True
    )

    if not ds_store_files:
        logger.warning("No .DS_Store files found")
        return

    for file in ds_store_files:
        os.remove(file)
        logger.info(f"Deleted {file}")

    logger.info(".DS_Store files deleted")


def process_parquet_files(folder_path, videos_folder_path=None):
    """
    Process all parquet files in the given folder by correcting the episode_index column.
    The value in episode_index will match the episode number in the filename.

    Args:
        folder_path (str): Path to the folder containing parquet files
        videos_folder_path (str): Path to the folder containing video files, if None, will not rename the video files
    """
    logger.info("Processing parquet files...")
    parquet_files = glob.glob(os.path.join(folder_path, "episode_*.parquet"))

    if not parquet_files:
        logger.warning(f"No parquet files found in {folder_path}")
        return

    logger.info(f"Found {len(parquet_files)} parquet files to process")

    # Order the files by episode number in ascending order
    parquet_files.sort(
        key=lambda x: int(re.search(r"episode_(\d+)\.parquet", x).group(1))
    )

    # Check if the episode number is continuous, if not, rename the parquet files and the corresponding videos
    episode_numbers = [
        int(re.search(r"episode_(\d+)\.parquet", file).group(1))
        for file in parquet_files
    ]

    # Make sure the list is ordered
    episode_numbers.sort()

    # Get names of the video folders in video_path
    if videos_folder_path is not None:
        video_folder = os.listdir(videos_folder_path)

    if episode_numbers != list(range(len(episode_numbers))):
        logger.warning(
            "Episode numbers are not continuous or starting from 0. Renaming files and videos..."
        )
        for i, file in enumerate(parquet_files):
            # We always start from 0
            new_episode_number = i
            new_file = os.path.join(
                folder_path, f"episode_{new_episode_number:06d}.parquet"
            )
            os.rename(file, new_file)
            logger.info(f"Renamed {file} to {new_file}")

            # Rename the corresponding video files
            if videos_folder_path is not None:
                for folder in video_folder:
                    new_video_file = os.path.join(
                        videos_folder_path, folder, f"episode_{new_episode_number:06d}.mp4"
                    )
                    video_file = os.path.join(
                        videos_folder_path, folder, f"episode_{episode_numbers[i]:06d}.mp4"
                    )
                    os.rename(video_file, new_video_file)
                    logger.info(f"Renamed {video_file} to {new_video_file}")

        # Update the list of parquet files after renaming
        parquet_files = glob.glob(os.path.join(folder_path, "episode_*.parquet"))
        parquet_files.sort(
            key=lambda x: int(re.search(r"episode_(\d+)\.parquet", x).group(1))
        )
        logger.info("Updated parquet files list after renaming")

    # Process each parquet file
    total_index = 0
    for file_path in parquet_files:
        # Extract episode number from filename using regex
        filename = os.path.basename(file_path)
        match = re.search(r"episode_(\d+)\.parquet", filename)

        if match:
            episode_number = int(match.group(1))
            logger.info(f"Processing {filename} - Episode {episode_number}")

            try:
                # Read the parquet file
                df = pd.read_parquet(file_path, engine="pyarrow")

                # Add episode_index column with the extracted number
                df["episode_index"] = episode_number

                # Rewrite frame_index column to go from 0 to n-1
                df["frame_index"] = range(len(df))

                # Rewrite index column to be a rolling index
                df["index"] = range(total_index, total_index + len(df))
                total_index += len(df)

                # If action is in degrees, convert to radians
                if max(df["action"].iloc[0][:6]) > 5.0:
                    logger.warning(f"Converting action to radians for {filename}")
                    # Convert each action array in the column
                    df["action"] = df["action"].apply(lambda action_array: np.array([
                        *np.deg2rad(action_array[:6]),  # Convert first 6 elements to radians
                        action_array[6]/10000,          # Scale element 6
                        *np.deg2rad(action_array[7:13]), # Convert elements 7-12 to radians
                        action_array[13]/10000          # Scale element 13
                    ]))

                # Save the modified DataFrame back to the same file
                df.to_parquet(file_path, index=False)

                logger.info(f"Successfully updated {filename}")

            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                sys.exit(1)
        else:
            logger.warning(f"Skipping {filename} - doesn't match expected pattern")

    logger.info("Parquet processing complete")


def run_stats_script(dataset_path):
    """Run the lerobot_stats_compute.py script with uv, fallback to python"""
    script_path = "lerobot_stats_compute.py"

    logger.info("Running lerobot_stats_compute.py...")

    try:
        subprocess.run(
            ["python", script_path, "--dataset-path", dataset_path, '--version', version],
            check=True,
        )
        logger.info(f"Successfully executed {script_path} with uv")

    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing {script_path}: {str(e)}")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"Execution error: {str(e)}")
        sys.exit(1)


def delete_indexes(indexes: list[int]):
    # Delete the parquet files
    logger.info("Deleting parquet files...")
    for parquets_folder_path in parquets_folder_paths:
        parquet_files = glob.glob(os.path.join(parquets_folder_path, "*.parquet"))
        for index in indexes:
            for file in parquet_files:
                if f"episode_{index:06d}.parquet" in file:
                    os.remove(file)
                    logger.info(f"Deleted file {file}")

    # Delete the corresponding video files
    if videos_folder_paths is not None:
        logger.info("Deleting video files...")
        video_folders = os.listdir(videos_folder_path)
        for index in indexes:
            for folder in video_folders:
                video_files = glob.glob(
                    os.path.join(videos_folder_path, folder, f"episode_{index:06d}.mp4")
                )
                for video_file in video_files:
                    os.remove(video_file)
                    logger.info(f"Deleted file {video_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error(
            "Usage: uv run repair_dataset.py --dataset-path <dataset_path> --indexes-to-delete <indexes_to_delete_comma_separated> --videos <True/False>\nFor example: python repair_dataset.py --dataset-path /path/to/dataset --indexes-to-delete 2,7,12 --videos True"
        )
        sys.exit(1)

    # Parse arguments using tyro
    config = tyro.cli(Config)

    # dataset_path is the parent folder of the parquet files
    dataset_path = config.dataset_path
    parquets_folder_paths = [os.path.join(dataset_path, "data", f"{chunk}") for chunk in os.listdir(os.path.join(dataset_path, "data"))]
    if config.videos: 
        videos_folder_paths = [os.path.join(dataset_path, "videos", f"{chunk}") for chunk in os.listdir(os.path.join(dataset_path, "videos"))]
    else:
        videos_folder_paths = None
    info_file_path = os.path.join(dataset_path, "meta", "info.json")

    # Check that the dataset is in the v2.0 or v2.1 format
    version = check_version(info_file_path)
    logger.info(f"Dataset version: {version}")

    # Add message and ask the user to press enter to continue
    message = ""
    if config.indexes_to_delete and config.dataset_path:
        message = f"This script will delete the following episodes: {config.indexes_to_delete} and repair the rest of your dataset: {config.dataset_path}.\nPress enter to continue..."
    else:
        message = f"This script will attempt to repair your dataset {config.dataset_path}.\nPress enter to continue..."

    logger.info(message)
    input()

    # indexes to delete
    if config.indexes_to_delete is not None:
        indexes_to_delete = config.indexes_to_delete.split(",")
        logger.info(f"Indexes to delete: {indexes_to_delete}")

        delete_indexes([int(index) for index in indexes_to_delete])

    # Delete all the .DS_Store files in the dataset
    delete_DS_Store(dataset_path)

    # Process parquet files
    if config.videos:
        for parquets_folder_path, videos_folder_path in zip(parquets_folder_paths, videos_folder_paths):
            process_parquet_files(parquets_folder_path, videos_folder_path)
    else:
        for parquets_folder_path in parquets_folder_paths:
            process_parquet_files(parquets_folder_path)

    # Run the stats script
    run_stats_script(dataset_path)

    # Push dataset to HF
    # print("Pushing dataset to HF...")
    # dataset_name = os.path.basename(dataset_path)
    # subprocess.run(
    #     [
    #         "uv",
    #         "run",
    #         "push_dataset_to_hf.py",
    #         dataset_path,
    #         dataset_name,
    #     ],
    #     check=True,
    # )
    # print("Dataset pushed successfully")
