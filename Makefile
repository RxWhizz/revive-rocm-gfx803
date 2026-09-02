# revive-rocm-gfx803 reproducible wrapper. Each target fails when its script fails.
SHELL := /bin/bash
.PHONY: help audit build build-base build-rocblas build-pytorch shell test test-gpu0 test-gpu1 test-dual \
        benchmark matmul-ab sgemm-ab gemm-shapes mace-test detect report clean-image

help:           ## show available targets
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_-]+:.*##/ {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

audit:          ## Phase 0 - host audit and gate
	bash scripts/audit_host.sh

build-base:     ; bash scripts/build_image.sh base
build-rocblas:  ; bash scripts/build_image.sh rocblas
build-pytorch:  ; bash scripts/build_image.sh pytorch
build:          ## Phase 5 - complete runtime image
	bash scripts/build_image.sh runtime

shell:          ## enter the container
	bash scripts/run_container.sh bash

detect:         ## print GPU info through PyTorch/HIP
	bash scripts/run_container.sh python tests/test_torch_info.py

test:           ## Phase 6 - torch info, matmul, backward, adam
	bash scripts/run_container.sh bash -lc 'for t in test_torch_info test_matmul test_backward test_optimizer; do echo "== $$t =="; python tests/$$t.py || exit 1; done'

test-gpu0:      ; HIP_VISIBLE_DEVICES=0 bash scripts/run_container.sh python tests/test_gpu_isolation.py
test-gpu1:      ; HIP_VISIBLE_DEVICES=1 bash scripts/run_container.sh python tests/test_gpu_isolation.py

test-dual:      ## Phase 6 - two processes, one per GPU
	bash scripts/run_dual_jobs.sh "python tests/test_optimizer.py"

benchmark:      ## Phase 7 - 30 minute dual stress test
	bash scripts/run_dual_jobs.sh "python tests/benchmark_gpu.py --minutes 30"

matmul-ab:      ## A/B rocBLAS/PyTorch FP32 matmul on both GPUs
	bash scripts/run_matmul_ab_dual.sh

sgemm-ab:       ## A/B standalone HIP SGEMM kernels on both GPUs
	bash scripts/run_sgemm_ab_dual.sh

gemm-shapes:    ## capture real MACE rocBLAS GEMM shapes with ROCBLAS_LAYER=2
	bash scripts/capture_gemm_shapes.sh

mace-test:      ## Phase 8 - MACE import, inference, tiny training
	bash scripts/run_container.sh bash -lc 'for t in test_mace_import test_mace_inference test_mace_training; do echo "== $$t =="; python mace/$$t.py || exit 1; done'

report:         ## print collected logs and summaries
	bash scripts/collect_logs.sh

clean-image:    ; sg docker -c "docker rmi $$(grep IMAGE_NAME environment/versions.env | cut -d= -f2):$$(grep IMAGE_TAG environment/versions.env | cut -d= -f2)" || true
