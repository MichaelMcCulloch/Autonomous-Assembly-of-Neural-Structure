FROM python:3.12-slim

RUN apt-get update && apt-get install -y swig build-essential && rm -rf /var/lib/apt/lists/*

RUN pip install \
    "gymnasium[box2d]>=1.2.0" \
    "loguru>=0.7.3" \
    "matplotlib>=3.10.3" \
    "numpy>=2.3.1" \
    "pandas>=2.3.1" \
    "pytest>=8.4.1" \
    "pyyaml>=6.0.2" \
    "rustworkx>=0.16.0" \
    "scipy>=1.16.0" \
    "seaborn>=0.13.2" \
    "tabulate>=0.9.0" 

RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.7.1"
