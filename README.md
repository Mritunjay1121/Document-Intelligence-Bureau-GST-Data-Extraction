# Document Intelligence: Bureau & GST Data Extraction

An AI-powered FastAPI service that extracts structured financial parameters from Bureau Credit Reports and GST Returns using GPT-4 Vision with RAG (Retrieval-Augmented Generation). The system converts PDF documents into clean JSON output with high accuracy and includes comprehensive testing and validation capabilities.

## Features

- **Vision-Based Extraction**: Uses GPT-4 Vision to extract data from complex PDF documents
- **RAG-Enhanced Accuracy**: Retrieval-Augmented Generation with domain-specific context for improved extraction
- **Batch Optimization**: Processes all 15 bureau parameters per page in a single API call
  - **14 API calls** for typical 14 paged document
- **Image Caching**: DPI 200 image caching for optimal performance
- **Clean JSON Output**: Structured output with value, source, and confidence per parameter
- **Comprehensive Testing**: Built-in accuracy testing and evaluation framework
- **Production Ready**: Handles file conflicts, cleanup, and error recovery

## Prerequisites

- Python 3.10 or higher
- pip for dependency management
- OpenAI API key (for GPT-4 Vision)
- Required Python packages (see requirements.txt)

## 🧠 Architecture & Flow

### High-Level Architecture

```
User Upload (PDFs)
     ↓
File Processing & Cleanup
     ↓
PDF → Images (PyMuPDF, DPI 200)
     ↓
[Vision Extraction with RAG]
     ├─ Parameter Definitions (Excel)
     ├─ Domain Knowledge (Embeddings)
     └─ GPT-4 Vision (Batch Processing)
     ↓
RAG Fallback (for missing parameters)
     ↓
Clean JSON Output
```

---

### 1. Document Upload & Processing

- User uploads Bureau PDF and GST PDF via API
- System performs automatic cleanup:
  - Deletes old files from uploads folder
  - Saves new files with proper naming
  - Prevents file mismatch issues
- Validates file formats and sizes

---

### 2. PDF to Image Conversion

- Convert PDF pages to high-quality images using PyMuPDF
- DPI 200 for optimal balance of quality and size
- Images cached for reuse across parameters
- Typical 14-page bureau report = 14 images

---

### 3. Batch Vision Extraction (Optimized)

**Key Innovation: Batch Processing**

Instead of iterating through all pages for each parameter:
```
For each page → Extract all parameters (14 pages × 1 call = 14 calls)
```

**Per-Page Process:**
- Send page image to GPT-4 Vision once
- Request all 15 bureau parameters simultaneously
- Include parameter descriptions and RAG context
- Parse JSON response for all found parameters

---

### 4. RAG-Enhanced Context

For each extraction:

**Domain Knowledge Retrieval:**
- Embed parameter descriptions
- Search domain knowledge base
- Retrieve relevant context snippets

**Context Injection:**
- Parameter-specific examples
- Common patterns and formats
- Edge case handling

Example for "bureau_credit_score":
```
Context: "Credit score typically 300-900. Found as 'CIBIL Score', 
'CRIF Score', or 'Bureau Score'. Usually in header or summary section."
```

---

### 5. Vision API Call Structure

**Single Batch Request:**
```python
{
  "model": "gpt-4o",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {...}},
        {"type": "text", "text": "Extract these 15 parameters: ..."}
      ]
    }
  ]
}
```

**Response Format:**
```json
{
  "bureau_credit_score": {
    "value": 627,
    "source": "Page 1, CRIF HM Score section",
    "confidence": 0.95,
    "context": "PERFORM CONSUMER 2.2 score"
  },
  "bureau_total_accounts": {
    "value": 54,
    "source": "Page 1, Account Summary",
    "confidence": 0.92,
    "context": "Number of Accounts"
  }
}
```

---

### 6. RAG Fallback for Missing Parameters

If vision extraction misses parameters:
- Use RAG pipeline to search entire document
- Extract text from all pages
- Semantic search for relevant sections
- GPT-4 extraction with focused context

**Fallback Process:**
```
Vision Result: Parameter not found
     ↓
Extract full PDF text
     ↓
RAG: Find relevant sections
     ↓
GPT-4: Extract from text
     ↓
Add to final results
```

---

### 7. GST Sales Extraction

**Monthly Sales Data:**
- Extract sales for last 12 months
- Parse GSTR-3B format automatically
- Handle multiple periods in single document

**Output Format:**
```json
{
  "gst_sales": [
    {
      "month": "January 2025",
      "sales": 951381,
      "source": "Section 3.1(a)"
    }
  ]
}
```

---

### 8. Final Output Structure

**Complete Response:**
```json
{
  "bureau": {
    "bureau_credit_score": {
      "value": 627,
      "source": "Page 1, CRIF HM Score",
      "confidence": 0.95
    },
    "bureau_total_accounts": {
      "value": 54,
      "source": "Page 1, Account Summary",
      "confidence": 0.92
    }
  },
  "gst_sales": [
    {
      "month": "January 2025",
      "sales": 951381,
      "source": "Section 3.1(a)"
    }
  ],
  "confidence_score": 0.87,
  "processing_time": "1.8 minutes"
}
```

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/bureau-gst-extraction.git
cd bureau-gst-extraction
```

### 2. Install and create a UV Python Package Manager

#### Installation

macos/linux
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
windows
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Install the requirements and activate the Environment

#### Install the requirements

```bash
uv add -r requirements.txt
```
#### Activate the environment


#### On macOS / Linux

```
source .venv/bin/activate
```

#### On windows

```
.venv\Scripts\activate
```


**Key Dependencies:**
- FastAPI
- PyMuPDF (fitz)
- OpenAI
- Sentence Transformers
- FAISS
- Pydantic
- python-multipart

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY="sk-your-key-here"
API_HOST="127.0.0.1"
API_PORT="8000"
```

### 5. Prepare Data Files

Place the parameter definition Excel file:

```
storage/
  ├── parameters/
  │   └── Bureau-parameters-Report.xlsx
  ├── uploads/
  │   ├── document_001.pdf
  │   └── document_002.pdf
  ├── gst_returns/
  │   ├── GSTR3B_06AAICK4577H1Z8_012025.pdf
  │   └── GSTR3B_06AAICK4577H1Z8_022025.pdf
  └── bureau_reports/
      ├── JEET_ARORA_PARK251217CR671901414.pdf
      └── JOHN_DOE_EXPERIAN_20250115.pdf

```

The Excel should contain:
- Parameter ID
- Parameter Name
- Description

### 7. Run the API

```bash
python main.py
```

The API will start at:

- **Swagger UI**: http://127.0.0.1:8000/docs
- **Health check**: http://127.0.0.1:8000/health

### Or use Uvicorn directly

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

---

## API Endpoints

### ROOT /

**Response:**

```json
{
  "message": "Document Intelligence: Bureau & GST Data Extraction v1.0",
  "version": "1.0.0",
  "status": "operational",
  "features": {
    "vision_extraction": true,
    "vision_model": "gpt-4o",
    "text_model": "gpt-4o-mini",
    "domain_knowledge": true
  },
  "endpoints": {
    "main": "POST /generate-rule",
    "health": "GET /health",
    "docs": "GET /docs"
  },
  "upload_required": "Both bureau_pdf and gst_pdf files are REQUIRED"
}
```

---

### POST /generate-rule

Extract parameters from Bureau and GST PDFs.

```
http://127.0.0.1:8000/generate-rule
```

**Request (Multipart Form Data):**

```
bureau_pdf: <file> (JEET_ARORA.pdf)
gst_pdf: <file> (GSTR3B_012025.pdf)
```

**Using CURL:**

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/generate-rule' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'bureau_pdf=@JEET_ARORA.pdf' \
  -F 'gst_pdf=@GSTR3B_012025.pdf'
```

**Response:**

```json
{
  "bureau": {
    "bureau_credit_score": {
      "value": 627,
      "source": "CRIF Report – Score Section"
    },
    "bureau_ntc_accepted": {
      "value": null,
      "status": "not_found"
    },
    "bureau_overdue_threshold": {
      "value": 5312886,
      "source": "Account Information, Row 25"
    },
    "bureau_dpd_30": {
      "value": null,
      "status": "not_found"
    },
    "bureau_dpd_60": {
      "value": null,
      "status": "not_found"
    },
    "bureau_dpd_90": {
      "value": null,
      "status": "not_found"
    },
    "bureau_settlement_writeoff": {
      "value": "0",
      "source": "Account Information Table - Settlement Amt column, Rows 1-5"
    },
    "bureau_no_live_pl_bl": {
      "value": null,
      "status": "not_found"
    },
    "bureau_suit_filed": {
      "value": false,
      "source": "Account Information - Account Remarks, Row 3"
    },
    "bureau_wilful_default": {
      "value": null,
      "status": "not_found"
    },
    "bureau_written_off_debt_amount": {
      "value": "0",
      "source": "Account Information Table - Total Writeoff Amt column, Rows 1-5"
    },
    "bureau_max_loans": {
      "value": null,
      "status": "not_found"
    },
    "bureau_loan_amount_threshold": {
      "value": 423000000,
      "source": "Account Information, Account 15, Collateral/Security Details"
    },
    "bureau_credit_inquiries": {
      "value": 13,
      "source": "Additional Summary - NUM-GRANTORS"
    },
    "bureau_max_active_loans": {
      "value": null,
      "status": "not_found"
    }
  },
  "gst_sales": [
    {
      "month": "January 2025",
      "sales": 951381,
      "source": "GSTR-3B Table 3.1(a)"
    }
  ],
  "confidence_score": 0.8
}
```

**Key Fields:**

- `bureau`: All bureau parameters with values, sources, and confidence
- `gst_sales`: Monthly sales data from GST returns
- `confidence_score`: Overall confidence (0-1)

---

### GET /health

Service health check.

**Response:**

```json
{
  "status": "healthy",
  "services": {
    "openai_client": true,
    "document_parser": true,
    "embedding_service": true,
    "rag_pipeline": true,
    "domain_rag": true,
    "vision_parser": true
  },
  "configuration": {
    "vision_enabled": true,
    "vision_model": "gpt-4o"
  }
}
```

---


## Testing & Accuracy Evaluation

The system includes a comprehensive testing framework to measure extraction accuracy.

### Setup Testing


1. **Configure ground truth:**

Edit `ground_truth.py` with expected values for your test documents:

```python
GROUND_TRUTH_BUREAU = {
    "YOUR_TEST_FILE.pdf": {
        "bureau_credit_score": {
            "expected_value": 627,
            "value_type": "number"
        },
        "bureau_total_accounts": {
            "expected_value": 54,
            "value_type": "number"
        }
        # ... more parameters
    }
}
```

### Run Tests

**Quick Test (2 runs):**

Before running this test.py you must configure the paths of "bureau_pdf" and "gst_pdf" files and "num_runs" inside this test.py . And also create a folder named 'test_data' in root directory and paste the bureau and gst files inside it . Then run the tests.

```bash
python test.py
```


### Test Output

```
================================================================================
RUNNING 100 EXTRACTIONS
================================================================================

Completed 100 extractions!

EVALUATING CONSISTENCY
────────────────────────────────────────────────────────────────────────────

Parameter: bureau_credit_score
  Total extractions: 100
  Unique values: 1
  Most common value: 627 (100/100 = 100.0%)
  ✅ 100% consistent

EVALUATING ACCURACY AGAINST GROUND TRUTH
────────────────────────────────────────────────────────────────────────────

✅ bureau_credit_score
   Expected: 627
   Extracted: 627
   Consistency: 100.0%

✅ bureau_total_accounts
   Expected: 54
   Extracted: 54
   Consistency: 100.0%

ACCURACY SUMMARY
────────────────────────────────────────────────────────────────────────────

Total parameters: 15
Correct: 14 (93.3%)
Incorrect: 0 (0.0%)
Missing: 1 (6.7%)

Overall Accuracy: 93.33%

✅ Report saved to: quick_test_report.json
```
