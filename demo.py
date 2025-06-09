# run_detection_workflow.py
import cv2
from inference import InferencePipeline

# ─── CONFIG ───────────────────────────────────────────────────────────────────
API_KEY        = "Bg7PeInJakWgMqsR7S1w"
WORKSPACE_NAME = "visionary-project-a56hi"      
WORKFLOW_ID    = "custom-workflow-2" 

def display_frame(result, video_frame):

    img = result["label_visualization"].numpy_image

    cv2.imshow("Live Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pipeline.stop()  # clean shutdown on 'q'

    print(result)

pipeline = InferencePipeline.init_with_workflow(
    api_key        = API_KEY,
    workspace_name = WORKSPACE_NAME,
    workflow_id    = WORKFLOW_ID,
    video_reference= 0,     
    max_fps        = 30,     
    on_prediction  = display_frame
)

pipeline.start()  
pipeline.join()  