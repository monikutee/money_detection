from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import io
import numpy as np
import cv2

from inference import get_model
import supervision as sv

# Inicializuojame FastAPI aplikaciją
app = FastAPI()

# Įkeliame modelį
model = get_model(model_id="geldbetrage-erkennen/2")

@app.post("/annotate")
async def annotate_image(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Konvertuojame baitus į numpy masyvą ir dekoduojame paveikslėlį
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Nepavyko atidaryti pateikto paveikslėlio.")
    
    # Vykdome inference
    results = model.infer(image)[0]
    detections = sv.Detections.from_inference(results)
    
    # Sukuriame anotavimo įrankius
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    # Anotuojame paveikslėlį
    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    
    # Užkoduoime anotetą paveikslėlį į JPEG formatą
    success, encoded_image = cv2.imencode('.jpg', annotated_image)
    if not success:
        raise HTTPException(status_code=500, detail="Nepavyko užkoduoti paveikslėlio.")
    
    # Grąžiname paveikslėlį kaip srautą
    return StreamingResponse(io.BytesIO(encoded_image.tobytes()), media_type="image/jpeg")