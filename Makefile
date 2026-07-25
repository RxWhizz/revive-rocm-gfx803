# revive-rocm-gfx803 — envoltura reproducible. No oculta errores (cada target falla si el script falla).
SHELL := /bin/bash
.PHONY: audit build build-base build-rocblas build-pytorch shell test test-gpu0 test-gpu1 test-dual \
        benchmark matmul-ab sgemm-ab gemm-shapes mace-test detect report clean-image

audit:          ## Fase 0 — auditoría host + gate
	bash scripts/audit_host.sh

build-base:     ; bash scripts/build_image.sh base
build-rocblas:  ; bash scripts/build_image.sh rocblas
build-pytorch:  ; bash scripts/build_image.sh pytorch
build:          ## Fase 5 — imagen runtime completa
	bash scripts/build_image.sh runtime

shell:          ## entrar al contenedor
	bash scripts/run_container.sh bash

detect:         ## info GPUs (torch_info)
	bash scripts/run_container.sh python tests/test_torch_info.py

test:           ## Fase 6 — torch info/matmul/backward/adam
	bash scripts/run_container.sh bash -lc 'for t in test_torch_info test_matmul test_backward test_optimizer; do echo "== $$t =="; python tests/$$t.py || exit 1; done'

test-gpu0:      ; HIP_VISIBLE_DEVICES=0 bash scripts/run_container.sh python tests/test_gpu_isolation.py
test-gpu1:      ; HIP_VISIBLE_DEVICES=1 bash scripts/run_container.sh python tests/test_gpu_isolation.py

test-dual:      ## Fase 6 — dos procesos, uno por GPU
	bash scripts/run_dual_jobs.sh "python tests/test_optimizer.py"

benchmark:      ## Fase 7 — estrés 30 min por GPU (dual)
	bash scripts/run_dual_jobs.sh "python tests/benchmark_gpu.py --minutes 30"

matmul-ab:      ## A/B rocBLAS/PyTorch FP32 matmul, ambas GPUs, CSV/JSON
	bash scripts/run_matmul_ab_dual.sh

sgemm-ab:       ## A/B kernels HIP propios: conservador/LDS/float4, ambas GPUs
	bash scripts/run_sgemm_ab_dual.sh

gemm-shapes:    ## Captura formas rocBLAS reales de MACE con ROCBLAS_LAYER=2
	bash scripts/capture_gemm_shapes.sh

mace-test:      ## Fase 8 — import/inferencia/entrenamiento mínimo
	bash scripts/run_container.sh bash -lc 'for t in test_mace_import test_mace_inference test_mace_training; do echo "== $$t =="; python mace/$$t.py || exit 1; done'

report:         ## recolecta logs/resultados
	bash scripts/collect_logs.sh

clean-image:    ; sg docker -c "docker rmi $$(grep IMAGE_NAME environment/versions.env | cut -d= -f2):$$(grep IMAGE_TAG environment/versions.env | cut -d= -f2)" || true
