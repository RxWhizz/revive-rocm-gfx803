# GEMM FP32 en RX 570 gfx803

## Diagnostico actual

- El stack ya ve 2 GPUs y `HIP_VISIBLE_DEVICES=0/1` aisla un proceso por tarjeta.
- El rendimiento observado de `torch.matmul`/rocBLAS es ~1.25 TFLOPS por RX 570 en `4096x4096x4096`.
- Pico teorico por tarjeta: `32 CU * 64 lanes * 2 FMA * 1.244 GHz = 5.09 TFLOPS`.
- 1.25 TFLOPS es ~24.5% del pico. Para GEMM grande esto apunta a kernel/logic suboptimo mas que a ancho de banda puro.
- El build de rocBLAS genero objetos gfx803, pero el log muestra seleccion desde logic vieja tipo `asm_full/r9nano` y `hip` mas fallback. Eso es compatible, no necesariamente size-tuned para Polaris RX570.

## Comprobar si usa ambas GPUs

Un solo proceso PyTorch no suma automaticamente las dos RX 570. Este proyecto usa un proceso por GPU:

```bash
make detect
make test-gpu0
make test-gpu1
make test-dual
make matmul-ab
```

`make matmul-ab` lanza dos procesos simultaneos, uno con `HIP_VISIBLE_DEVICES=0` y otro con `HIP_VISIBLE_DEVICES=1`, y escribe:

- `results/matmul_ab_gpu0.json`
- `results/matmul_ab_gpu1.json`
- `results/matmul_ab_summary.txt`

Si ambos JSON existen y el resumen muestra dos GPUs, el benchmark dual realmente esta usando ambas tarjetas.

## Saber si limita memoria, ocupacion, bloque o sincronizacion

Usa las dos capas de benchmark:

```bash
make matmul-ab
make sgemm-ab
```

Interpretacion rapida:

- Si `torch_rocblas` supera claramente a `lds`, el cuello esta en seleccion/tuning de rocBLAS para tus formas, no en la capacidad basica de la GPU.
- Si `lds` mejora mucho contra `conservative`, el kernel naive estaba limitado por memoria global y reutilizacion pobre.
- Si `vec4` mejora solo cuando `N` es multiplo de 4, hay beneficio de coalescing/vector loads; si no mejora, el limite esta en ocupacion, registros o reutilizacion de `B`.
- Si todos se quedan bajos en matrices grandes, revisar clocks/temperatura/power limit con `rocm-smi`, y validar que no haya throttle.

## Variantes incluidas

Archivo: `benchmarks/sgemm_gfx803_hip.cpp`.

1. `conservative`
   - Un thread calcula un elemento de `C`.
   - Bloque `16x16`.
   - Sirve como baseline de correctitud y muestra el coste de leer `A/B` desde memoria global sin reutilizacion.

2. `lds`
   - Tile `16x16` en LDS/shared memory.
   - Reutiliza tiles de `A` y `B` dentro del workgroup.
   - Debe mejorar sobre `conservative` en matrices medianas/grandes si el problema era trafico de memoria global.

3. `vec4`
   - Un thread calcula cuatro columnas contiguas usando `float4` para `B`.
   - Ayuda cuando `N` esta alineado a 4 y la matriz esta en layout contiguo.
   - Es util para medir si las cargas vectorizadas/coalescidas ayudan en las formas reales.

Compilacion manual dentro del contenedor:

```bash
bash scripts/build_sgemm_kernels.sh
/workspace/results/sgemm_gfx803_hip --shape 4096 --repeats 20 --warmup 5 --csv /workspace/results/sgemm_ab.csv
```

## Capturar formas GEMM reales de MACE/ALIGNN

Tuning sin formas reales desperdicia horas. Captura primero:

```bash
make gemm-shapes
```

Por defecto corre:

```bash
ROCBLAS_LAYER=2 python mace/test_mace_inference.py
```

Salidas:

- `results/gemm_shapes_gpu0.log`: log bruto rocBLAS.
- `results/gemm_shapes_gpu0.csv`: distribucion `(m,n,k,transA,transB,batch_count)`.
- `results/gemm_shapes_gpu0_summary.txt`: top formas por frecuencia.

Para otro workload:

```bash
GPU=0 bash scripts/capture_gemm_shapes.sh "python mace/test_mace_training.py"
```

## Tuning Tensile

Plantilla inicial:

```bash
configs/tensile_gfx803_sgemm_template.yaml
```

Flujo recomendado:

1. Capturar formas reales con `make gemm-shapes`.
2. Copiar las formas dominantes desde `results/gemm_shapes_gpu*.csv` a `BenchmarkFinalParameters`.
3. Ejecutar `Tensile` en hardware gfx803 dentro del entorno de build de rocBLAS.
4. Instalar la logic resultante en el arbol de rocBLAS o construir una libreria aparte.
5. Reconstruir rocBLAS/PyTorch runtime y repetir `make matmul-ab`.

La meta no es optimizar `4096^3` si MACE/ALIGNN usa GEMMs flacos. La meta es maximizar el area bajo la distribucion real de formas.
