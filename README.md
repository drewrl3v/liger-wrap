# liger-wrap

This repo serves as a proof of concept wrapper to simply integrate the Liger Kernel modules into your pytorch models.

## Requirements:

## First:

    uv pip install torch --index-url=https://download.pytorch.org/whl/cu126
    uv pip install ninja cmake wheel pybind11 setuptools numpy

## Second:

In a separate working directory clone triton.

    git clone https://github.com/triton-lang/triton.git
    cd triton
    git checkout release/3.2.x
    uv pip install -e python # install triton
    uv pip install liger-kernel