import cv2
import os
from pathlib import Path

def extract_frames_from_video(video_path, output_folder="data", frame_prefix="frame", frame_step=5):
    """
    Extract frames from a video every N frames and save them as images in the specified folder.

    Args:
        video_path (str): Path to the input video file
        output_folder (str): Folder to save extracted frames (default: "data")
        frame_prefix (str): Prefix for frame filenames (default: "frame")
        frame_step (int): Extract every Nth frame (default: 5)

    Returns:
        int: Number of frames extracted
    """

    # Create output directory if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return 0

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video: {video_path}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Extracting every {frame_step} frames to: {output_folder}/")

    frame_count = 0
    extracted_count = 0

    while True:
        # Read frame from video
        ret, frame = cap.read()

        if not ret:
            break

        # Extract frame only if it's at the specified step interval
        if frame_count % frame_step == 0:
            # Create filename with zero-padded frame number
            frame_filename = f"{frame_prefix}_{extracted_count:06d}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)

            # Save frame as image
            cv2.imwrite(frame_path, frame)
            extracted_count += 1

            # Print progress every 20 extracted frames
            if extracted_count % 20 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {frame_count}/{total_frames} frames processed ({progress:.1f}%) - {extracted_count} frames extracted")

        frame_count += 1

    # Release video capture
    cap.release()

    print(f"✅ Extraction complete! {extracted_count} frames saved to {output_folder}/")
    print(f"Processed {frame_count} total frames, extracted every {frame_step} frames")
    return extracted_count

def extract_frames_at_interval(video_path, output_folder="data", interval_seconds=1.0, frame_prefix="frame"):
    """
    Extract frames from a video at specified time intervals.

    Args:
        video_path (str): Path to the input video file
        output_folder (str): Folder to save extracted frames (default: "data")
        interval_seconds (float): Time interval between extracted frames in seconds
        frame_prefix (str): Prefix for frame filenames (default: "frame")

    Returns:
        int: Number of frames extracted
    """

    # Create output directory if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return 0

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video: {video_path}")
    print(f"FPS: {fps:.2f}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Extracting frames every {interval_seconds} seconds to: {output_folder}/")

    frame_interval = int(fps * interval_seconds)
    frame_count = 0
    extracted_count = 0

    while True:
        # Read frame from video
        ret, frame = cap.read()

        if not ret:
            break

        # Extract frame if it's at the specified interval
        if frame_count % frame_interval == 0:
            # Create filename with timestamp
            timestamp = frame_count / fps
            frame_filename = f"{frame_prefix}_{timestamp:.2f}s_{extracted_count:04d}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)

            # Save frame as image
            cv2.imwrite(frame_path, frame)
            extracted_count += 1

            print(f"Extracted frame at {timestamp:.2f}s -> {frame_filename}")

        frame_count += 1

    # Release video capture
    cap.release()

    print(f"✅ Extraction complete! {extracted_count} frames saved to {output_folder}/")
    return extracted_count

def draw_time_stats_overlay(frame, time_stats, frame_count, fps):
    """
    Draw time tracking statistics overlay on the video frame.

    Args:
        frame: Video frame to draw on
        time_stats: Dictionary with time statistics
        frame_count: Current frame number
        fps: Video FPS for time calculation
    """
    # Calculate current video time
    current_time = frame_count / fps

    # Position for the stats box (top-right corner) - Much bigger now
    box_x = frame.shape[1] - 650  # 650 pixels from right edge
    box_y = 20
    box_width = 620
    box_height = 320

    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Draw border
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 255, 255), 2)

    # Text properties - Much bigger text
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_font_scale = 1.5
    main_font_scale = 1.2
    thickness = 3
    line_spacing = 45

    # Title - Much bigger
    cv2.putText(frame, "FIGHT TIME BREAKDOWN", (box_x + 20, box_y + 45),
                font, title_font_scale, (255, 255, 255), thickness)

    # Current time - Bigger
    cv2.putText(frame, f"Time: {current_time:.1f}s", (box_x + 20, box_y + 95),
                font, main_font_scale, (255, 255, 255), thickness)

    # Standing time (Green) - Bigger
    standing_time = time_stats.get('standing', 0)
    cv2.putText(frame, f"Standing: {standing_time:.1f}s", (box_x + 20, box_y + 140),
                font, main_font_scale, (0, 255, 0), thickness)

    # Clinch time (Yellow) - Bigger
    clinch_time = time_stats.get('clinch', 0)
    cv2.putText(frame, f"Clinch: {clinch_time:.1f}s", (box_x + 20, box_y + 185),
                font, main_font_scale, (0, 255, 255), thickness)

    # Ground time (Red) - Bigger
    ground_time = time_stats.get('ground', 0)
    cv2.putText(frame, f"Ground: {ground_time:.1f}s", (box_x + 20, box_y + 230),
                font, main_font_scale, (0, 0, 255), thickness)

    # Calculate percentages
    total_time = standing_time + clinch_time + ground_time
    if total_time > 0:
        standing_pct = (standing_time / total_time) * 100
        clinch_pct = (clinch_time / total_time) * 100
        ground_pct = (ground_time / total_time) * 100

        # Draw percentage bar - Much bigger
        bar_x = box_x + 20
        bar_y = box_y + 260
        bar_width = 400
        bar_height = 30

        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)

        # Standing portion (Green)
        standing_width = int(bar_width * (standing_pct / 100))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + standing_width, bar_y + bar_height), (0, 255, 0), -1)

        # Clinch portion (Yellow)
        clinch_start = bar_x + standing_width
        clinch_width = int(bar_width * (clinch_pct / 100))
        cv2.rectangle(frame, (clinch_start, bar_y), (clinch_start + clinch_width, bar_y + bar_height), (0, 255, 255), -1)

        # Ground portion (Red)
        ground_start = clinch_start + clinch_width
        ground_width = int(bar_width * (ground_pct / 100))
        cv2.rectangle(frame, (ground_start, bar_y), (ground_start + ground_width, bar_y + bar_height), (0, 0, 255), -1)

        # Percentage labels - Bigger and better positioned
        cv2.putText(frame, f"Standing: {standing_pct:.0f}%", (box_x + 450, box_y + 150),
                    font, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Clinch: {clinch_pct:.0f}%", (box_x + 450, box_y + 195),
                    font, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Ground: {ground_pct:.0f}%", (box_x + 450, box_y + 240),
                    font, 0.8, (0, 0, 255), 2)

def create_annotated_video_from_csv(video_path, csv_path, output_path="annotated_video.mp4", csv_fps=30):
    """
    Create an annotated video with bounding boxes from CSV batch predictions.

    Args:
        video_path (str): Path to the input video file
        csv_path (str): Path to the CSV file with predictions
        output_path (str): Path for the output annotated video
        csv_fps (int): FPS that the CSV predictions were generated at (default: 30)

    Returns:
        bool: True if successful, False otherwise
    """
    import pandas as pd
    import json

    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} prediction rows from CSV")

        # Open input video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return False

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Create output video writer with original FPS
        out = cv2.VideoWriter(output_path, fourcc, original_fps, (width, height))

        print(f"Creating annotated video: {output_path}")
        print(f"Video dimensions: {width}x{height}")
        print(f"Original video FPS: {original_fps:.2f}")
        print(f"CSV predictions FPS: {csv_fps}")
        print(f"FPS ratio: {csv_fps / original_fps:.2f}")

        # Initialize time tracking
        frame_count = 0
        time_stats = {'standing': 0.0, 'clinch': 0.0, 'ground': 0.0}
        frame_duration = 1.0 / original_fps  # Time per frame in seconds

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate which CSV prediction row corresponds to this video frame
            # This accounts for FPS differences between video and CSV
            csv_frame_index = int(frame_count * (csv_fps / original_fps))

            # Get predictions for this frame (if available)
            if csv_frame_index < len(df):
                predictions_str = df.iloc[csv_frame_index]['predictions']

                try:
                    # Parse the JSON string
                    predictions_data = json.loads(predictions_str)
                    predictions = predictions_data.get('predictions', [])

                    # Track time for each class detected in this frame
                    frame_classes = set()

                    # Draw bounding boxes and labels
                    for pred in predictions:
                        # Extract bounding box coordinates (center format)
                        x_center = pred['x']
                        y_center = pred['y']
                        box_width = pred['width']
                        box_height = pred['height']

                        # Convert to corner coordinates
                        x1 = int(x_center - box_width / 2)
                        y1 = int(y_center - box_height / 2)
                        x2 = int(x_center + box_width / 2)
                        y2 = int(y_center + box_height / 2)

                        # Get class and confidence
                        class_name = pred['class']
                        confidence = pred['confidence']

                        # Track which classes are present in this frame
                        if 'Standing' in class_name:
                            frame_classes.add('standing')
                            color = (0, 255, 0)  # Green for standing
                        elif 'Clinch' in class_name:
                            frame_classes.add('clinch')
                            color = (0, 255, 255)  # Yellow for clinch
                        elif 'Ground' in class_name:
                            frame_classes.add('ground')
                            color = (0, 0, 255)  # Red for ground
                        else:
                            color = (255, 255, 255)  # White for unknown

                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # Draw label with confidence
                        label = f"{class_name}: {confidence:.2f}"
                        font_scale = 1.0  # Increased from 0.6
                        thickness = 3     # Increased from 2
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

                        # Draw label background
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 15),
                                    (x1 + label_size[0] + 10, y1), color, -1)

                        # Draw label text
                        cv2.putText(frame, label, (x1 + 5, y1 - 8),
                                  cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

                    # Update time statistics for detected classes
                    for class_type in frame_classes:
                        time_stats[class_type] += frame_duration

                except json.JSONDecodeError as e:
                    print(f"Error parsing predictions for video frame {frame_count} (CSV row {csv_frame_index}): {e}")

            # Draw time tracking overlay
            draw_time_stats_overlay(frame, time_stats, frame_count, original_fps)

            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_count + 1}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            # Write frame to output video
            out.write(frame)
            frame_count += 1

            # Print progress every 100 frames
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")

        # Release everything
        cap.release()
        out.release()

        print(f"✅ Annotated video created successfully!")
        print(f"Output: {output_path}")
        print(f"Total frames processed: {frame_count}")

        return True

    except Exception as e:
        print(f"Error creating annotated video: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    video_file = "testVideoSwitch.mp4"  # Change this to your video file
    csv_file = "testVideoSwitch.csv"  # Change this to your CSV file

    print("Choose option:")
    print("1. Extract ALL frames")
    print("2. Extract frames at intervals")
    print("3. Create annotated video from CSV")

    choice = input("Enter choice (1, 2, or 3): ").strip()

    if choice == "1":
        extract_frames_from_video(video_file)
    elif choice == "2":
        interval = float(input("Enter interval in seconds (e.g., 1.0): "))
        extract_frames_at_interval(video_file, interval_seconds=interval)
    elif choice == "3":
        create_annotated_video_from_csv(video_file, csv_file, "annotated_ufc_fight.mp4", csv_fps=30)
    else:
        print("Invalid choice!")