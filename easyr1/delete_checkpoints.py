import os
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import re

class CheckpointHandler(FileSystemEventHandler):
    def __init__(self, folder_path, max_checkpoints=2):
        self.folder_path = folder_path
        self.max_checkpoints = max_checkpoints

    def on_created(self, event):
        if not event.is_directory:
            return
        # No need to call cleanup_checkpoints here if we're already calling it every 30 minutes

    def cleanup_checkpoints(self):
        # List all subdirectories in the folder
        checkpoints = [os.path.join(self.folder_path, d) for d in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, d))]
        
        # Filter checkpoints that match the pattern "checkpoint-<number>"
        checkpoints = [checkpoint for checkpoint in checkpoints if re.match(r'global_step_\d+', os.path.basename(checkpoint))]

        # Get creation time and sort by creation time
        checkpoints_with_time = [(os.path.getctime(checkpoint), checkpoint) for checkpoint in checkpoints]
        checkpoints_with_time.sort()  # Sort by creation time
        
        specific_checkpoints = {f"global_step_{i}" for i in [45, 90, 135, 180, 220]}  # Add more as needed

        # Remove all but the last max_checkpoints directories
        if len(checkpoints_with_time) <= self.max_checkpoints:
            print(f"No need to remove any checkpoints, {len(checkpoints_with_time)} checkpoints exist")
        else:
            for _, checkpoint in checkpoints_with_time[:-self.max_checkpoints]:
                checkpoint_name = os.path.basename(checkpoint)
                if checkpoint_name not in specific_checkpoints:
                    shutil.rmtree(checkpoint)
                    print(f"Removed old checkpoint: {checkpoint}")
                else:
                    print(f"Skipped specific checkpoint: {checkpoint}")

def main():
    folder_path = '/data/wuxinrui/easyr1_checkpoints/1_5B_TCMv2_long_short_regular_budget_modified'  # Change this to your path
    event_handler = CheckpointHandler(folder_path)
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=False)
    observer.start()

    try:
        while True:
            event_handler.cleanup_checkpoints()  # Call cleanup_checkpoints every 30 minutes
            time.sleep(300)  # Wait for 5 minutes
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()