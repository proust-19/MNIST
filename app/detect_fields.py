import logging
import re
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

DATE_PATTERN = re.compile(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})")
AMOUNT_PATTERN = re.compile(
    r"(?:total\s+amount|grand\s+total|amount\s+due|total|amount|due|grand|sum)"
    r"[:\s]*[₹rRs\.]*\s*([\d,]+\.?\d*)"
)
GST_PATTERN = re.compile(r"(\d{2}[A-Za-z]{5}\d{4}[A-Za-z][A-Za-z\d]Z[A-Za-z\d])")
INVOICE_PATTERN = re.compile(r"(?:invoice|inv|bill)\s*(?:no|#|number|\.)?[:\s]*(\S+)")

VENDOR_SKIP_KEYWORDS = {
    "invoice",
    "date",
    "gst",
    "tax",
    "bill",
    "address",
    "phone",
    "email",
    "www",
    "http",
    "total",
    "amount",
    "contact",
    "tel",
    "mobile",
    "page",
    "ref",
    "our",
    "your",
    "ship",
    "sold",
    "buyer",
}


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        pil = Image.open(path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.fastNlMeansDenoising(thresh, h=30)


def ocr_image(img: np.ndarray) -> str:
    config = "--oem 3 --psm 4"
    return pytesseract.image_to_string(img, config=config, lang="eng")


def clean_vendor_name(name: str) -> str:
    name = name.strip().strip("\"'")
    name = re.sub(r"\bPvtLid\b", "Pvt Ltd", name, flags=re.IGNORECASE)
    name = re.sub(r"\bPvt\.?Ltd\.?\b", "Pvt Ltd", name, flags=re.IGNORECASE)
    return name


def extract_fields(text: str) -> dict:
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    invoice_no = None
    date = None
    total = None
    gst = None
    vendor = None

    for line in lines:
        line_lower = line.lower()

        inv_match = INVOICE_PATTERN.search(line_lower)
        if inv_match and invoice_no is None:
            invoice_no = inv_match.group(1)

        date_match = DATE_PATTERN.search(line)
        if date_match and date is None:
            date = date_match.group(1)

        amount_match = AMOUNT_PATTERN.search(line_lower)
        if amount_match and total is None:
            try:
                total = float(amount_match.group(1).replace(",", ""))
            except ValueError:
                pass

        gst_match = GST_PATTERN.search(line, re.IGNORECASE)
        if gst_match and gst is None:
            gst = gst_match.group(1).upper()

    likely_name_lines = [
        l
        for l in lines[:6]
        if 5 < len(l) < 80
        and not re.search(r"\d{4,}", l)
        and not any(kw in l.lower() for kw in VENDOR_SKIP_KEYWORDS)
        and not re.match(r"^\d", l)
    ]
    if likely_name_lines:
        vendor = clean_vendor_name(likely_name_lines[0])

    return {
        "vendor_name": vendor or "",
        "invoice_number": invoice_no or "",
        "date": date or "",
        "total_amount": total if total else 0.0,
        "gst_number": gst or "",
        "raw_text_length": len(text),
    }


def find_digit_regions(img: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 10 < w < 100 and 10 < h < 100 and 0.3 < w / h < 1.5:
            regions.append(thresh[y : y + h, x : x + w])
    return regions


def detect_fields(path: Path) -> dict:
    try:
        img = load_image(path)
    except Exception as e:
        logger.error("Failed to load image: %s", e)
        raise

    processed = preprocess_for_ocr(img)
    text = ocr_image(processed)
    fields = extract_fields(text)
    fields["regions_found"] = len(find_digit_regions(img))
    return fields
