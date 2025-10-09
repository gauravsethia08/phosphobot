#!/usr/bin/env python3

"""
This script will process all parquets files in the specified folder by correcting the episode_index column.
We then compute the statistics using the lerobot_stats_compute.py script.
"""

import os
import re
import sys
import glob
import cv2
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


def convert_bytes_to_numpy(image_bytes):
    """
    Convert bytes to numpy array.

    Args:
        image_bytes (bytes): The image in bytes format.

    Returns:
        np.ndarray: The image as a numpy array in BGR format.
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Decode the numpy array to an image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def process_parquet_files(folder_path, dataset_path):
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

                # Delete 1st row; temporarily fixes an issue with some datasets
                # #if len(df) > 1:
                # logger.info(f"DF Len before - {len(df)}")
                # df = df.iloc[1:].reset_index(drop=True)
                # logger.info(f"DF Len after - {len(df)}")

                # # Add episode_index column with the extracted number
                # df["episode_index"] = episode_number

                # # Rewrite frame_index column to go from 0 to n-1
                # df["frame_index"] = range(len(df))

                # # Rewrite index column to be a rolling index
                # df["index"] = range(total_index, total_index + len(df))
                # total_index += len(df)

                # # If action is in degrees, convert to radians
                # if max(df["action"].iloc[0][:6]) > 5.0:
                #     logger.warning(f"Converting action to radians for {filename}")
                #     # Convert each action array in the column
                #     df["action"] = df["action"].apply(lambda action_array: np.array([
                #         *np.deg2rad(action_array[:6]),  # Convert first 6 elements to radians
                #         action_array[6]/10000,          # Scale element 6
                #         *np.deg2rad(action_array[7:13]), # Convert elements 7-12 to radians
                #         action_array[13]/10000,
                #         *action_array[14:],         # Scale element 13
                #     ]))

                # # Save the modified DataFrame back to the same file
                # df.to_parquet(file_path, index=False)

                # Store the videos
                chunk_num = file_path.split("/")[-2]
                video_path = os.path.join(dataset_path, "videos", chunk_num) 
                os.makedirs(video_path, exist_ok=True)

                camera_views = ["observation.images.cam_high", "observation.images.cam_low", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
                import concurrent.futures

                def save_video_for_view(view):
                    video_filename = f"episode_{episode_number:06d}.mp4"
                    video_file_path = os.path.join(video_path, view.split('.')[-1], video_filename)
                    if os.path.exists(video_file_path):
                        logger.info(f"Video file exists: {video_file_path}")
                        return
                    else:
                        logger.warning(f"Video file does not exist: {video_file_path}")
                        video_writer = None
                        for idx, row in df.iterrows():
                            frame = row[view]
                            frame_numpy = convert_bytes_to_numpy(frame['bytes'])
                            height, width, layers = frame_numpy.shape
                            if idx == 0:
                                video_writer = cv2.VideoWriter(video_file_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
                            video_writer.write(frame_numpy)
                        if video_writer:
                            video_writer.release()
                        logger.info(f"Saved video: {video_file_path}")

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    executor.map(save_video_for_view, camera_views)

                logger.info(f"Successfully updated {filename}")

            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                sys.exit(1)
        else:
            logger.warning(f"Skipping {filename} - doesn't match expected pattern")

    logger.info("Parquet processing complete")



if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error(
            "Usage: uv run save_videos.py --dataset-path <dataset_path> \nFor example: python save_videos.py --dataset-path /path/to/dataset"
        )
        sys.exit(1)

    # Parse arguments using tyro
    config = tyro.cli(Config)

    # dataset_path is the parent folder of the parquet files
    dataset_path = config.dataset_path
    parquets_folder_paths = [os.path.join(dataset_path, "data", f"{chunk}") for chunk in os.listdir(os.path.join(dataset_path, "data"))]
    
    videos_folder_path = os.path.join(dataset_path, "videos")
    os.makedirs(videos_folder_path, exist_ok=True)
    videos_chunks_path = [os.path.join(videos_folder_path, f"{chunk}") for chunk in os.listdir(os.path.join(dataset_path, "data"))]
    [os.makedirs(chunk_path, exist_ok=True) for chunk_path in videos_chunks_path]

    # For each chunk make dirs for the each camera view
    camera_views = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
    for chunk_path in videos_chunks_path:
        for view in camera_views:
            os.makedirs(os.path.join(chunk_path, view), exist_ok=True)

    # Loop over all the parquet files and store the videos as MP4 files
    for parquets_folder_path in parquets_folder_paths:
            process_parquet_files(parquets_folder_path, dataset_path)
    # process_parquet_files(parquets_folder_paths, dataset_path)