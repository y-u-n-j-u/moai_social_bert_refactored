FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /workspace

RUN python -m pip uninstall --yes ninja \
    && python -m pip install --no-cache-dir \
      numpy==1.23.5 \
      pandas==1.5.3 \
      scipy==1.10.1 \
      PyYAML==6.0.2 \
      tqdm==4.67.1 \
      Pillow==10.4.0 \
      matplotlib==3.7.5 \
      opencv-python-headless==4.10.0.84 \
      dill==0.3.8 \
      transformers==4.48.3 \
    && python -m pip check

COPY . .

RUN python -m compileall -q . \
    && python -c "from src.utils import bootstrap_paths; bootstrap_paths(); import torch, transformers, numpy, pandas, scipy, yaml, tqdm, PIL, cv2, matplotlib, dill, sbert, spubert; print('imports OK', torch.__version__, transformers.__version__)"

CMD ["python", "scripts/smoke_test_gazebo.py"]
