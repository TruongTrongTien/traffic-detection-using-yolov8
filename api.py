import io
from fastapi import responses, FastAPI, UploadFile
from PIL import Image
from configs import *
from main import *
import tempfile

app = FastAPI()

class CFG:
    image_size = IMAGE_SIZE
    conf_thres = 0.5
    iou_thres = 0.3
cfg = CFG()

def detect_image(image):
    pred = prediction(session=load_session(PATH_MODEL),
                           image=image,
                           cfg=cfg)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return visualize(image, pred)

@app.post("/detect/")
async def detect(file: UploadFile):
    if file.content_type and file.content_type.startswith("image/"):
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_numpy = np.array(image)
        image_detected = detect_image(image_numpy)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image_detected.save(temp_file, format="JPEG")
            temp_file_path = temp_file.name
        return responses.FileResponse(temp_file_path, media_type="image/jpeg")
    
        # image_bytes = io.BytesIO()
        # image_detected.save(image_bytes, format="JPEG")
        # image_bytes.seek(0)
        # return responses.StreamingResponse(content=io.BytesIO(image_detected.tobytes()), media_type="image/jpeg")
    