from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.detect_fields import detect_fields

app = FastAPI(title="DocScanner - Document Field Extraction")


class ExtractionResult(BaseModel):
    vendor_name: str
    invoice_number: str
    date: str
    total_amount: float
    gst_number: str
    raw_text_length: int
    regions_found: int


SUPPORTED = {".png", ".jpg", ".jpeg", ".pdf", ".tiff", ".bmp"}


@app.get("/health")
def health():
    return {"status": "ok", "service": "DocScanner"}


@app.post("/extract", response_model=ExtractionResult)
async def extract(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Unsupported: {suffix}")

    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        fields = detect_fields(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return ExtractionResult(**fields)
