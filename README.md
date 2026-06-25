# GenAI Streamlit Suite

Two Streamlit apps powered by Google Gemini in one repository:

- **MedScan**: upload a medical image and get a structured clinical-style analysis.
- **GenAI SQL Helper**: turn plain English into a SQL query with an explanation.



## Project Structure

```text
.
├── medscan_app.py
└── sqlquery_app.py
```

## Features

### MedScan

- Upload PNG, JPG, JPEG, or WEBP images.
- Add optional patient history.
- Uses Gemini to generate a structured analysis.

### SQL Helper

- Enter a request in plain English.
- Generates a SQL query.
- Returns a short explanation and example usage.

## Requirements

- Python 3.10 or newer
- Streamlit
- `google-generativeai`
- `Pillow` for the MedScan app

## Setup

1. Create and activate a virtual environment.
2. Install the dependencies:

```bash
pip install streamlit google-generativeai pillow
```

## Run the Apps

### MedScan

```bash
streamlit run medscan_app.py
```

### SQL Helper

```bash
streamlit run sqlquery_app.py
```
