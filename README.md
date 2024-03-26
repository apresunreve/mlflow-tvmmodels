# TVM models with MLFlow

# Requirements
Only needs to install tvm runtime:
`pip install apache-tvm`


# Files
`defonet_soybean_tvm_models/`: TVM models  in .so format.

`save_tvm_model_to_mlflow.py`: Example script to save tvm models to MLFlow format

 

## Example usage 
```
python save_tvm_model_to_mlflow.py \
--file="defonet_soybean_tvm_models/1705674234.6740797.best.pth_tuned20000.so" \
--res=108 \
--target="llvm -mcpu=cortex-a72" \
--save_model_path="tvm_defonet_corn_mlflow"
```
The script will save models in MLFlow format.  We have already run this script and the output files are stored at `defonet_soybean_mlflow_models/`.
