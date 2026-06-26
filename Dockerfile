# CAID inference container. This doesnt include functions to generate embedings, data or testing, only inference

FROM python:3.11-slim

# CPU-only PyTorch 
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir 'numpy>=1.23' 'pyyaml>=6.0' 'biopython>=1.80'

WORKDIR /app
COPY linker_core/ ./linker_core/
COPY methods/ ./methods/
COPY predict_caid.py ./

# CLI entrypoint; pass predict_caid.py args after the image name.
ENTRYPOINT ["python", "predict_caid.py"]
