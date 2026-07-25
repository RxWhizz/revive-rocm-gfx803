# revive-pytorch-gfx803 — ROCm 5.7.1 + rocBLAS(gfx803, Tensile completo) + PyTorch 2.2.2 (fuente, gfx803)
# Multi-stage: base → rocblas → pytorch → runtime.  Build largo (rocBLAS Tensile gfx803 = horas).
# syntax=docker/dockerfile:1.6

############################ STAGE base ############################
FROM rocm/dev-ubuntu-22.04:5.7.1 AS base
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTORCH_ROCM_ARCH=gfx803 \
    AMDGPU_TARGETS=gfx803 \
    GPU_TARGETS=gfx803 \
    HCC_AMDGPU_TARGET=gfx803 \
    TORCH_BLAS_PREFER_HIPBLASLT=0 \
    ROCM_PATH=/opt/rocm \
    PATH=/opt/rocm/bin:/opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates build-essential cmake ninja-build ccache \
        python3 python3-dev python3-pip python3-venv pkg-config \
        libnuma-dev gfortran libomp-dev \
    && rm -rf /var/lib/apt/lists/*
# (la imagen rocm/dev ya trae hipcc + LLVM-amdgpu; no se necesita rocm-llvm-dev)

# micromamba + entorno fijado (Python 3.10) — gestiona deps Python (no otra ROCm)
ARG MAMBA_VERSION=1.5.8
RUN curl -Ls "https://micro.mamba.pm/api/micromamba/linux-64/${MAMBA_VERSION}" \
      | tar -xvj -C /usr/local bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda
COPY environment/environment.yml /tmp/environment.yml
RUN micromamba create -y -p /opt/conda -f /tmp/environment.yml && micromamba clean -ay

############################ STAGE rocblas (gfx803, Tensile completo) ############################
FROM base AS rocblas
# CRÍTICO: sacar conda del PATH para el build de rocBLAS. Tensile usa joblib/loky y lanza workers
# con el primer 'python3' del PATH; con conda primero usaba el python de conda (sin joblib) → fallo.
# El build usa el system python (joblib vía apt python3-joblib) + el virtualenv que crea install.sh.
ENV PATH=/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
WORKDIR /build
ENV CCACHE_DIR=/ccache
# rocBLAS de la rama que empareja ROCm 5.7. install.sh -a gfx803 genera la Tensile library gfx803.
RUN git clone --depth 1 -b release/rocm-rel-5.7 https://github.com/ROCm/rocBLAS.git
# apt-get update primero: base borró /var/lib/apt/lists; install.sh -d hace apt-get install sin update.
# Sin tee: el exit code de install.sh debe propagarse (el log lo captura el redirect externo del build).
RUN --mount=type=cache,target=/ccache \
    apt-get update && \
    cd rocBLAS && ./install.sh -d -a gfx803 -i
# Resultado: paquete .deb + librería en /opt/rocm; copiamos el .deb para runtime
RUN mkdir -p /artifacts && ( cp -v rocBLAS/build/release/*.deb /artifacts/ 2>/dev/null || \
    find rocBLAS/build -name '*.deb' -exec cp -v {} /artifacts/ \; )

############################ STAGE pytorch (2.2.2, fuente, gfx803) ############################
FROM rocblas AS pytorch
WORKDIR /build
ENV CMAKE_PREFIX_PATH=/opt/conda \
    USE_ROCM=1 USE_CUDA=0 USE_NCCL=0 USE_DISTRIBUTED=0 \
    BUILD_TEST=0 USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 \
    MAX_JOBS=24
# Clone PRIMERO (capa cacheada; cambiar deps/libs/build no re-clona los cientos de submódulos)
RUN git clone --depth 1 -b v2.2.2 --recursive https://github.com/pytorch/pytorch.git
# Librerías ROCm que PyTorch (LoadHIP.cmake) busca y la imagen dev no trae. apt-mark hold protege
# NUESTRO rocBLAS gfx803 (3.1.0-b80e422) para que hipblas-dev no lo reemplace por el stock sin gfx803.
RUN apt-get update && apt-mark hold rocblas rocblas-dev && apt-get install -y --no-install-recommends \
        rocrand-dev rocprim-dev hipcub-dev rocthrust-dev rocsparse-dev hipsparse-dev \
        rocfft-dev hipfft-dev miopen-hip-dev rccl-dev rocsolver-dev hipsolver-dev hipblas-dev \
    && rm -rf /var/lib/apt/lists/*
# deps de build. cmake==3.26.4: PyTorch 2.2 usa submódulos con cmake_minimum_required<3.5 que CMake 4.x
# RECHAZA (la 3.26 aún los soporta). ninja para acelerar. (sin dataclasses: es stdlib en py3.10.)
RUN micromamba run -p /opt/conda pip install --no-cache-dir \
        pyyaml typing_extensions numpy==1.26.4 "cmake==3.26.4" ninja setuptools wheel \
        requests mkl-include
# Hipify (CUDA→HIP) + compila el wheel gfx803. Sin tee: el error real debe propagarse.
RUN --mount=type=cache,target=/ccache \
    cd pytorch && \
    micromamba run -p /opt/conda python tools/amd_build/build_amd.py && \
    PYTORCH_ROCM_ARCH=gfx803 USE_ROCM=1 TORCH_BLAS_PREFER_HIPBLASLT=0 \
      micromamba run -p /opt/conda python setup.py bdist_wheel && \
    mkdir -p /artifacts && cp -v dist/torch-*.whl /artifacts/

############################ STAGE runtime ############################
# Correctitud primero: FROM pytorch hereda rocBLAS gfx803 + libs ROCm + conda + el wheel ya en /artifacts.
# (Optimización futura: runtime lean desde rocm/runtime copiando solo libs .so necesarias.)
FROM pytorch AS runtime
ENV PATH=/opt/rocm/bin:/opt/conda/bin:$PATH \
    PYTORCH_ROCM_ARCH=gfx803 TORCH_BLAS_PREFER_HIPBLASLT=0 ROCM_PATH=/opt/rocm
# Instala el wheel de torch gfx803 + ASE + MACE en el entorno conda
RUN micromamba run -p /opt/conda pip install --no-cache-dir /artifacts/torch-*.whl && \
    micromamba run -p /opt/conda pip install --no-cache-dir ase==3.22.1 mace-torch==0.3.6
WORKDIR /workspace
COPY tests/ /workspace/tests/
COPY mace/ /workspace/mace/
COPY docker/entrypoint.sh docker/healthcheck.sh /workspace/
RUN chmod +x /workspace/entrypoint.sh /workspace/healthcheck.sh
ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["bash"]
