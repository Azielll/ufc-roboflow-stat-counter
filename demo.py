# run_detection_workflow.py
import cv2
from inference import InferencePipeline
import time

# ─── CONFIG ───────────────────────────────────────────────────────────────────
API_KEY        = "Bg7PeInJakWgMqsR7S1w"
WORKSPACE_NAME = "visionary-project-a56hi"      
WORKFLOW_ID    = "custom-workflow-3" 
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

def on_prediction(results, video_frame):
    global total_frames_processed, start_time

    predictions = results["predictions"]
    num_predictions = len(predictions)

    # Increment frame counter and calculate actual elapsed time
    total_frames_processed += 1
    elapsed_time = time.time() - start_time

    print(f"Frame {total_frames_processed} ({elapsed_time:.2f}s real-time) has {num_predictions} prediction(s):")

    for i, pred in enumerate(predictions):
        class_name = pred[5]['class_name']
        confidence = pred[2]
        print(f"  {i+1}. Class: {class_name} (confidence: {confidence:.3f})")

    print("----------")
    

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