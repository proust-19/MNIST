from app.detect_fields import extract_fields

INVOICE_TEXT = """
ABC Electronics Pvt Ltd
123 Industrial Area, Ahmedabad - 380001
GST: 24AABCE1234D1Z5

Invoice No: INV-2024-0042
Date: 15-06-2024

Description          Qty    Rate    Amount
Laptop              2     45000    90000
Mouse               5       500     2500

Total Amount: ₹92,500.00
"""


def test_extract_vendor():
    result = extract_fields(INVOICE_TEXT)
    assert "ABC Electronics" in result["vendor_name"]


def test_extract_invoice_number():
    result = extract_fields(INVOICE_TEXT)
    assert result["invoice_number"] == "inv-2024-0042"


def test_extract_date():
    result = extract_fields(INVOICE_TEXT)
    assert result["date"] == "15-06-2024"


def test_extract_total():
    result = extract_fields(INVOICE_TEXT)
    assert result["total_amount"] == 92500.0


def test_extract_gst():
    result = extract_fields(INVOICE_TEXT)
    assert result["gst_number"] == "24AABCE1234D1Z5"


def test_amount_compound_phrase():
    result = extract_fields("Total Amount: Rs. 92500.00")
    assert result["total_amount"] == 92500.0


def test_amount_grand_total():
    result = extract_fields("Grand Total: ₹1,500.00")
    assert result["total_amount"] == 1500.0


def test_clean_vendor_pvt_lid():
    result = extract_fields("ABC Electronics PvtLid\nInvoice No: 123")
    assert "Pvt Ltd" in result["vendor_name"]


def test_empty_text():
    result = extract_fields("")
    assert result["vendor_name"] == ""
    assert result["total_amount"] == 0.0
