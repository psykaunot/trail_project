#!/usr/bin/env python3
import os
import shutil
import time
import sys

def cleanup_perception_logs():
    """
    Cleans up the perception_logs directory:
    1. Removes the color_correction folder
    2. Removes edge_detection_* images
    3. Removes specialized_books_* images
    Keeps only yolo_books_* and yolo_objects_* images
    """
    print("Starting cleanup of perception logs...")
    
    # Path to the perception logs directory
    perception_logs_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "perception", "perception_logs"
    )
    
    if not os.path.exists(perception_logs_dir):
        print(f"Directory {perception_logs_dir} does not exist. Nothing to clean up.")
        return
    
    # Remove color_correction folder
    color_correction_dir = os.path.join(perception_logs_dir, "color_correction")
    if os.path.exists(color_correction_dir):
        print(f"Removing color_correction directory: {color_correction_dir}")
        shutil.rmtree(color_correction_dir)
    
    # Remove unwanted image files
    removed_count = 0
    for filename in os.listdir(perception_logs_dir):
        if filename.startswith(("edge_detection_", "specialized_books_")):
            file_path = os.path.join(perception_logs_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                removed_count += 1
    
    print(f"Removed {removed_count} unwanted image files")
    print("Cleanup complete!")

if __name__ == "__main__":
    cleanup_perception_logs()