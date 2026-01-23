# Qwen-Coder Response Annotation System

Online annotation interface for evaluating 4 different Qwen-Coder response approaches:
- Baseline NLQ
- RAG TF-IDF
- RAG Embedding
- RAG Hybrid

## Statistics
- Total Queries: 1542
- Responses per Query: 4
- Total Annotations Needed: 6168

## For Annotators

### Access the App
This app is deployed at: [Your Streamlit URL will be here]

### Instructions
1. Click the link above
2. Enter your name/ID
3. Review the query
4. Evaluate all 4 responses (tabs)
5. Click "Save & Next"
6. Export your annotations when done

### Time Estimate
- ~20-30 minutes per query
- ~10-15 queries per session recommended

## Deployment

This app is deployed on Streamlit Cloud.

### Local Testing
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Files
- `streamlit_app.py` - Main application
- `annotation_tasks.csv` - Annotation data  
- `requirements.txt` - Dependencies
- `README.md` - This file
