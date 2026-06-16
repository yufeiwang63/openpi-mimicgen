# path=/tmp
path="${PWD}"
export HF_LEROBOT_HOME="${path}/data"
export HF_HOME="${path}/data/cache/huggingface"
export HF_DATASETS_CACHE=$HF_HOME/datasets
export OPENPI_DATA_HOME="${path}/data/cache/openpi/"
export OPENPI_JAX_CACHE_DIR="${path}/data/cache/jax"
export UV_CACHE_DIR="${path}/data/cache/"
echo "HF_LEROBOT_HOME: $HF_LEROBOT_HOME"
echo "HF_HOME: $HF_HOME"
echo "HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "OPENPI_DATA_HOME: $OPENPI_DATA_HOME"
echo "OPENPI_JAX_CACHE_DIR: $OPENPI_JAX_CACHE_DIR"
echo "UV_CACHE_DIR: $UV_CACHE_DIR"
mkdir -p $HF_LEROBOT_HOME
mkdir -p $HF_HOME
mkdir -p $HF_DATASETS_CACHE
mkdir -p $OPENPI_DATA_HOME
mkdir -p $OPENPI_JAX_CACHE_DIR
mkdir -p $UV_CACHE_DIR