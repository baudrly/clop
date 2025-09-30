FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

LABEL org.opencontainers.image.source="https://github.com/${GITHUB_REPOSITORY}"

# Install additional dependencies
RUN pip install --upgrade pip && \
    pip install numpy pandas scikit-learn matplotlib seaborn fpdf \
                onnx onnxruntime polars

# Set working directory
WORKDIR /workspace

# Copy your script
COPY src/clop.py .

# Run with GPU support:
# docker build -t clop-cuda .
# docker run --gpus all -v $(pwd):/workspace clop-cuda python clop.py --input_file sampled_dataset3.fa
