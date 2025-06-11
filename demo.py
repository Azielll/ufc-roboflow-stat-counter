# run_detection_workflow.py
import cv2
from inference import InferencePipeline
import time

# ─── CONFIG ───────────────────────────────────────────────────────────────────
API_KEY        = "Bg7PeInJakWgMqsR7S1w"
WORKSPACE_NAME = "visionary-project-a56hi"      
WORKFLOW_ID    = "production3" 
INPUT_VIDEO    = "testVideoSwitch.mp4"
OUTPUT_VIDEO   = "output_fight_with_timers.mp4"

cap = cv2.VideoCapture(INPUT_VIDEO)
fps     = cap.get(cv2.CAP_PROP_FPS)
width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
writer  = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# Time tracking variables
import time
start_time = time.time()
total_frames_processed = 0
last_frame_time = start_time
current_state = None
state_timers = {"standing": 0.0, "ground": 0.0}

def update_timers(predictions, frame_duration):
    """Update both standing and ground timers based on what's detected"""
    global state_timers

    standing_detected = False
    ground_detected = False

    for pred in predictions:
        class_name = pred[5]['class_name'].lower()
        if 'standing' in class_name or 'clinch' in class_name:
            standing_detected = True
        elif 'ground' in class_name:
            ground_detected = True

    # Add time to both timers based on what's detected
    if standing_detected:
        state_timers["standing"] += frame_duration
    if ground_detected:
        state_timers["ground"] += frame_duration

def on_prediction(results, video_frame):
    global total_frames_processed, start_time, last_frame_time, state_timers

    predictions = results["predictions"]
    num_predictions = len(predictions)

    # Calculate time since last frame
    current_time = time.time()
    frame_duration = current_time - last_frame_time

    # Update timers based on what's detected in this frame
    update_timers(predictions, frame_duration)

    # Update tracking variables
    total_frames_processed += 1
    elapsed_time = current_time - start_time
    last_frame_time = current_time

    # Display the video frame
    # Extract the numpy array from VideoFrame object
    frame_copy = video_frame.image.copy()

    # Add timer overlay to the frame
    cv2.putText(frame_copy, f"Standing: {state_timers['standing']:.1f}s",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame_copy, f"Ground: {state_timers['ground']:.1f}s",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame_copy, f"Frame: {total_frames_processed}",
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow("UFC Fight Analysis", frame_copy)

    # Press 'q' to quit, any other key to continue
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        exit()

    print(f"Frame {total_frames_processed} ({elapsed_time:.2f}s real-time) has {num_predictions} prediction(s):")

    for i, pred in enumerate(predictions):
        class_name = pred[5]['class_name']
        confidence = pred[2]
        print(f"  {i+1}. Class: {class_name} (confidence: {confidence:.3f})")

    print(f"Fight Time Breakdown:")
    print(f"  Standing: {state_timers['standing']:.2f}s")
    print(f"  Ground: {state_timers['ground']:.2f}s")
    print("---")
    

pipeline = InferencePipeline.init_with_workflow(
    api_key        = API_KEY,
    workspace_name = WORKSPACE_NAME,
    workflow_id    = WORKFLOW_ID,
    video_reference= INPUT_VIDEO,     
    max_fps        = 30,     
    on_prediction  = on_prediction
)

pipeline.start()  
pipeline.join()  