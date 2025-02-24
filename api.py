from annotate_image import run_inference
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import torch
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
import numpy as np
import uvicorn
import io

app = FastAPI()

# Enable CORS
origins = [
    "*"
    # add other allowed origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows the defined origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["bill_count", "coin_count"]  # Expose custom headers

)


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Validate that the uploaded file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    try:
        # Read the uploaded image into memory
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run inference to get annotated image and counts
        annotated_image, bill_count, coin_count = run_inference(image)
        
        # Encode the annotated image as JPEG into memory
        success, encoded_image = cv2.imencode('.jpg', annotated_image)
        if not success:
            raise HTTPException(status_code=500, detail="Could not encode image")
        
        # Prepare the response as a streaming response with appropriate headers
        headers = {
            "bill_count": str(bill_count),
            "coin_count": str(coin_count)
        }
        print(bill_count,coin_count)
        return StreamingResponse(io.BytesIO(encoded_image.tobytes()),
                                 media_type="image/jpeg",
                                 headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
