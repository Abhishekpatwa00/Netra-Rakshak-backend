from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Netra Rakshak API")

# Enable CORS for website integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session = None
input_name = None

@app.on_event("startup")
def load_model():
    global session, input_name
    session = ort.InferenceSession("netra_rakshak.onnx")
    input_name = session.get_inputs()[0].name
    print(f"Loaded ONNX model. Input tensor name: {input_name}")

@app.get("/")
def read_root():
    return {"status": "API is online", "endpoint": "/predict"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 1. Resize image strictly to 224 x 224
        image = image.resize((224, 224), Image.Resampling.LANCZOS)

        # 2. Convert to Float32 array and normalize pixel values to [0, 1]
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # 3. Transpose from (Height, Width, Channels) [224, 224, 3] 
        #    to (Channels, Height, Width) [3, 224, 224]
        img_array = np.transpose(img_array, (2, 0, 1))
        
        # 4. Add batch dimension -> Shape: [1, 3, 224, 224]
        img_array = np.expand_dims(img_array, axis=0)

        # 5. Run prediction
        outputs = session.run(None, {input_name: img_array})
        predictions = outputs[0].tolist()

        return {
            "status": "success",
            "input_shape": list(img_array.shape),  # Will output [1, 3, 224, 224]
            "predictions": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))