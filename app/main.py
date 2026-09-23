from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from bs4 import BeautifulSoup
from pathlib import Path
import fitz
from PIL import Image
import pytesseract
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

app = FastAPI(title="Binary Sentiment Classifier")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
VECTORIZER_PATH = BASE_DIR / "vectorizer.joblib"
STOPWORDS = set(stopwords.words("english"))

model = None
vectorizer = None


class TextRequest(BaseModel):
    text: str


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def extract_pdf_text(path: str) -> str:
    try:
        doc = fitz.open(path)
        parts = [page.get_text("text") for page in doc]
        doc.close()
        return "\n".join(parts)
    except Exception:
        return ""


def extract_html_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return soup.get_text(" ")
    except Exception:
        return ""


def extract_image_text(path: str) -> str:
    try:
        img = Image.open(path).convert("RGB")
        return pytesseract.image_to_string(img)
    except Exception:
        return ""


def load_artifacts():
    global model, vectorizer
    if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    else:
        model = None
        vectorizer = None


def predict_label(text: str):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model artifacts not found. Run train.py first.")
    x = vectorizer.transform([clean_text(text)])
    pred = model.predict(x)[0]
    return "positive" if int(pred) == 1 else "negative"


@app.on_event("startup")
def startup_event():
    load_artifacts()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    }


@app.post("/predict")
def predict(payload: TextRequest):
    return {"label": predict_label(payload.text)}


@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    filename = file.filename.lower()
    content = await file.read()
    temp_path = f"/tmp/{file.filename}"

    with open(temp_path, "wb") as f:
        f.write(content)

    if filename.endswith(".pdf"):
        raw_text = extract_pdf_text(temp_path)
    elif filename.endswith((".html", ".htm")):
        raw_text = extract_html_text(temp_path)
    elif filename.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
        raw_text = extract_image_text(temp_path)
    else:
        raw_text = content.decode("utf-8", errors="ignore")

    cleaned = clean_text(raw_text)
    return {
        "label": predict_label(cleaned),
        "extracted_characters": len(raw_text),
        "cleaned_characters": len(cleaned)
    }

# Remove the old image
## docker images
## docker rmi -f sentiment-api

# Rebuild the image
## sudo docker build -t sentiment-api app/

# Run the container
## sudo docker run --rm --name sentiment-api sentiment-api

# run the API directly in Codespaces
## uvicorn main:app --host 0.0.0.0 --port 8000

# Test and Run
## curl http://127.0.0.1:8000/health
## curl -X POST "http://127.0.0.1:8000/predict" \
##   -H "Content-Type: application/json" \
##   -d '{"text":"I love this product"}'

# sudo docker stop sentiment-api
