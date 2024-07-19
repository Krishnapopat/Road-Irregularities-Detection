import asyncio
import cv2
import numpy as np
import supervision as sv
from fastapi import FastAPI, WebSocket, Request
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaStreamTrack
from av import VideoFrame
from ultralytics import YOLOv10
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import torch
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the index.html file at the root URL
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.get("/favicon.ico")
async def favicon():
    return FileResponse('static/favicon.ico')

pcs = set()
model = YOLOv10('best.pt')
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
executor = ThreadPoolExecutor(max_workers=4)

class MLVideoStreamTrack(VideoStreamTrack):
    def __init__(self, track):
        super().__init__()
        self.track = track
        self.frame_count = 0
        self.processed_frame = None  # Store the processed frame
        self.processing_task = None  # Store the async task for processing
        print("MLVideoStreamTrack initialized with track:", track)

    async def recv(self):
        self.frame_count += 1
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        if self.frame_count % 3 == 0:  # Process every 3rd frame
            if self.processing_task is None or self.processing_task.done():
                img_resized = cv2.resize(img, (320, 240))
                self.processing_task = asyncio.create_task(self.process_frame(img_resized))
        
        if self.processed_frame is not None:
            img = self.processed_frame

        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

    async def process_frame(self, img):
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(executor, self.run_inference, img)
        self.processed_frame = self.annotate_frame(img, results)

    def run_inference(self, img):
        with torch.no_grad():
            results = model(img)[0]
        return sv.Detections.from_ultralytics(results)

    def annotate_frame(self, img, results):
        annotated_image = bounding_box_annotator.annotate(scene=img, detections=results)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=results)
        return annotated_image

@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    print("Received offer:", params)  # Logging the offer for debugging
    offer = RTCSessionDescription(sdp=params["sdp"], type="offer")
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("track")
    def on_track(track):
        print("Track received:", track)  # Logging track information
        if track.kind == "video":
            pc.addTrack(MLVideoStreamTrack(track))
            print("Video track added to RTCPeerConnection")

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    print("Sending answer:", {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})  # Logging the answer
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

@app.post("/ice-candidate")
async def ice_candidate(request: Request):
    params = await request.json()
    print("Received ICE candidate:", params["candidate"])  # Logging ICE candidates
    return {"status": "received"}

@app.on_event("shutdown")
async def on_shutdown():
    print("Shutdown event triggered")
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    print("All RTCPeerConnections closed and pcs cleared")
if __name__ == "__main__":
    import uvicorn
    print("Starting server with uvicorn")
    uvicorn.run(app, host="0.0.0.0", port=8000, ssl_certfile="cert.pem", ssl_keyfile="key.pem")
