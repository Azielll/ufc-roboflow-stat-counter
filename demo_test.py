# Import the InferencePipeline object
from inference import InferencePipeline
import cv2

def my_sink(result, video_frame):
    
    img = result["output_image"].numpy_image

    cv2.imshow("Workflow Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pipeline.stop()
    
    print(result) # do something with the predictions of each frame


# initialize a pipeline object
pipeline = InferencePipeline.init_with_workflow(
    api_key="Bg7PeInJakWgMqsR7S1w",
    workspace_name="visionary-project-a56hi",
    workflow_id="production3",
    video_reference=0, # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
    max_fps=30,
    on_prediction=my_sink
)
pipeline.start() #start the pipeline
pipeline.join() #wait for the pipeline thread to finish
