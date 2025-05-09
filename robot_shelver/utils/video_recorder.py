import cv2
import os
import time
import threading
import numpy as np

class VideoRecorder:
# In video_recorder.py, update the __init__ method:

    def __init__(self, output_path=None, fps=30):
        """Initialize video recorder."""
        # Create recordings directory if it doesn't exist
        self.recordings_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "recordings"
        )
        os.makedirs(self.recordings_dir, exist_ok=True)

        # Set default output path with timestamp for uniqueness
        if output_path is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"habitat_recording_{timestamp}.mp4"

        # Ensure full path
        if not os.path.isabs(output_path):
            self.output_path = os.path.join(self.recordings_dir, output_path)
        else:
            self.output_path = output_path

        self.fps = fps
        self.is_recording = False
        self.writer = None
        self.record_thread = None
        self.frame_queue = []
        self.lock = threading.Lock()

        print(f"Video recorder initialized. Recordings will be saved to: {self.recordings_dir}")
    
    def start_recording(self):
        """Start recording video."""
        if self.is_recording:
            print("Already recording!")
            return False
        
        self.is_recording = True
        self.frame_queue = []
        self.record_thread = threading.Thread(target=self._record_thread, daemon=True)
        self.record_thread.start()
        print(f"Recording started! Output will be saved to {self.output_path}")
        return True
    
    def add_frame(self, frame):
        """Add a frame to the recording."""
        if not self.is_recording:
            return
        
        # Make a copy to prevent modifications
        with self.lock:
            self.frame_queue.append(frame.copy())
    
    def stop_recording(self):
        """Stop recording and save the video."""
        if not self.is_recording:
            print("Not recording!")
            return False
        
        self.is_recording = False
        if self.record_thread:
            self.record_thread.join(timeout=5.0)
        
        print(f"Recording stopped! Video saved to {self.output_path}")
        return True
    
    # In VideoRecorder's add_frame method:
    def add_frame(self, frame):
        """Add a frame to the recording."""
        if not self.is_recording:
            return
        
        # Make a copy to prevent modifications
        with self.lock:
            self.frame_queue.append(frame.copy())
            # Print status every 30 frames (~1 second at 30fps)
            if len(self.frame_queue) % 30 == 0:
                print(f"Recording: {len(self.frame_queue)} frames queued")
    

    def _record_thread(self):
        """Thread function to write video frames."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_size = None
        writer = None
        
        while self.is_recording or len(self.frame_queue) > 0:
            # Get a frame if available
            frame = None
            with self.lock:
                if self.frame_queue:
                    frame = self.frame_queue.pop(0)
            
            if frame is not None:
                # Initialize writer if needed
                if writer is None:
                    frame_size = (frame.shape[1], frame.shape[0])
                    writer = cv2.VideoWriter(
                        self.output_path, fourcc, self.fps, frame_size, True
                    )
                
                # Convert RGBA to RGB if needed
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                elif frame.shape[2] == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
                # Write the frame
                writer.write(frame)
            
            # Slight pause to avoid hogging CPU
            time.sleep(0.001)
        
        # Clean up
        if writer is not None:
            writer.release()