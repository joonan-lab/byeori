# The image that cuts a new paper's figures and tables.
#
# marker and its models are baked in because the alternative was to install them at task start,
# and this worker exists to run on one paper at a time: a paper converts in 4 to 14 seconds, and
# a three-minute pip install before each one would have forced papers to be batched and their
# figures to arrive hours after the paper did.
#
# What is deliberately absent is the VLM. marker's default path uses a vision model for layout
# and a second one for recognition, and surya serves those through llama.cpp on any machine
# without an NVIDIA GPU, which a Fargate task is. Setting disable_ocr switches the layout
# detector to the small rf-detr model and never constructs the VLM at all
# (marker/builders/layout.py:86-93), so this image holds two modest models instead of several
# gigabytes of weights it could not run anyway.
#
# The worker code itself is NOT baked in. It is downloaded from s3://{bucket}/worker/assets.py at
# start, the same way the extraction worker is, so fixing the extraction logic does not mean
# rebuilding and pushing an image.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

# libgl and libglib are what torchvision and PIL link against; they are not in the slim image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# torch first, from the CPU index. Left to itself, marker's install pulls the aarch64 torch from
# PyPI, which now depends on the whole CUDA stack - cuda-toolkit, cudnn, cublas, nccl, triton -
# for the ARM machines that have an NVIDIA GPU attached. This task has none, and that stack was
# 2.7 GB of an image Fargate pulls again for every paper. Installing torch from the CPU index
# first satisfies marker's requirement, so none of it is fetched.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && pip install --no-cache-dir marker-pdf==2.0.0 boto3 fpdf2

COPY infra/asset_warmup.py /opt/warmup.py

# The build converts a real PDF. If a model is missing, if the layout path reaches for the VLM,
# or if anything wants llama-server, the build fails here rather than in production.
RUN python /opt/warmup.py && rm -rf /root/.cache/pip

CMD ["sh", "-c", "python -c \"import boto3,os; boto3.client('s3').download_file(os.environ['BUCKET_NAME'], 'worker/assets.py', '/tmp/assets.py')\" && python /tmp/assets.py"]
