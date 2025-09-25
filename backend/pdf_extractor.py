"""
PDF Extractor for Cambridge IELTS Reading Tests
Simple script to extract text from PDF files
"""

import os
import sys
from pathlib import Path

def extract_text_simple(pdf_path):
    """Extract text from PDF using basic approach"""
    try:
        # Try PyPDF2 first
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    except ImportError:
        print("PyPDF2 not available")
    except Exception as e:
        print(f"PyPDF2 error: {e}")
    
    try:
        # Try pdfplumber
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
    except ImportError:
        print("pdfplumber not available")
    except Exception as e:
        print(f"pdfplumber error: {e}")
    
    try:
        # Try pymupdf
        import fitz  # pymupdf
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text
    except ImportError:
        print("pymupdf not available")
    except Exception as e:
        print(f"pymupdf error: {e}")
    
    return None

def test_pdf_extraction():
    """Test PDF extraction on a sample file"""
    # Find a sample PDF file
    cambridge_path = Path("/Users/shan/Desktop/Work/Projects/EdPrep AI/IELTS/Cambridge IELTS/Academic")
    
    if not cambridge_path.exists():
        print(f"Cambridge IELTS path not found: {cambridge_path}")
        return
    
    # Look for any PDF file
    pdf_files = list(cambridge_path.rglob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found")
        return
    
    # Test with the first PDF file
    test_pdf = pdf_files[0]
    print(f"Testing with: {test_pdf}")
    
    text = extract_text_simple(test_pdf)
    
    if text:
        print(f"✅ Successfully extracted {len(text)} characters")
        print(f"First 500 characters:\n{text[:500]}...")
        return text
    else:
        print("❌ Failed to extract text")
        return None

if __name__ == "__main__":
    test_pdf_extraction()
