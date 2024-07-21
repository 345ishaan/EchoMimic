import os
import logging
import uvicorn
import uuid
from pydantic import BaseModel

from fastapi import FastAPI

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
#from genime_creator import GenimeCreator
from typing import List
from supabase import create_client, Client
import sys
sys.path.append("/home/EchoMimic")
import genime_inference
import shutil

app = FastAPI()

SUPABASE_URL = "https://ttvaarlnqssopdguetwq.supabase.co/"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR0dmFhcmxucXNzb3BkZ3VldHdxIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcyMDcxODMwNSwiZXhwIjoyMDM2Mjk0MzA1fQ.TFoNnJhvghFsinuAjbNILN1ohJkz41vbv9y-4Eva12g"

supa_client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')

#genime_creator = GenimeCreator()


class LipSyncSingleRequest(BaseModel):
    video_url: str
    audio_url: str

class LipSyncRequest(BaseModel):
    reqs: List[LipSyncSingleRequest]



@app.get("/test")
async def root():
    return "Hello World"


@app.post("/create-lipsync")
async def create_lipsync(lipsyncReq: LipSyncRequest):
    """Endpoint for lipsync model inference.
    """

    save_dir = f"/home/EchoMimic/{str(uuid.uuid4())}"
    genime_inference.infer(image_urls=[
        req.video_url for req in lipsyncReq.reqs],
        audio_urls=[
        req.audio_url for req in lipsyncReq.reqs],
        save_dir=save_dir)
    save_fname = os.path.join(save_dir, 'final_concat.mp4')
    if not os.path.exists(save_fname):
        raise Exception("Could not retrieve final lip sync")
    with open(save_fname, 'rb') as f:
        supa_client.storage.from_("genime-bucket").upload(file=f,
                path=save_fname, 
                file_options={"content-type": "video/mp4"})
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    return save_fname


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
