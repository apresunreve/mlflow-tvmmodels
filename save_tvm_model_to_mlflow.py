import numpy as np
import mlflow.pyfunc
import tvm
from tvm.contrib import graph_executor
from sys import version_info

# Define the tvm model wrapper
class TVMWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, res=108, target="llvm", dtype="float32", input_name="input0"):
        self.target = target
        self.dtype = dtype
        self.res = res
        self.input_shape = (1,3,res,res)
        self.input_name = input_name


    def load_context(self, context):
        self.device = tvm.device(str(self.target), 0)

        # load from .so file
        lib = tvm.runtime.load_module(context.artifacts['tvm_executable'])
        self.module = graph_executor.GraphModule(lib["default"](self.device))


    def predict(self, context, model_input, params=None):
        # assuming the input is np.ndarray
        data_tvm = tvm.nd.array(model_input.astype(self.dtype))
        self.module.set_input(self.input_name, data_tvm)
        self.module.run()

        # get output
        out = self.module.get_output(0)
        return out


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
                    prog='SaveTVMModel',
                    description='Save tvm .so models to MLFlow format',
                    )
    parser.add_argument('-f', '--file', type=str,
                        help="tvm .so executable path",
                        default="defonet_soybean_tvm_models/1705674234.6740797.best.pth_tuned20000.so")

    parser.add_argument('-d', '--dtype', type=str,
                        help="data type for model input",
                        default="float32")

    parser.add_argument('-t', '--target', type=str,
                        help="tvm target device",
                        default="llvm -mcpu=cortex-a72")

    parser.add_argument('-r', '--res', type=int,
                        help="input image resolution",
                        default=108)

    parser.add_argument('-s', '--save_model_path', type=str,
                        help="Path to save MLFlow model",
                        default="")

    args = parser.parse_args()

    dtype = args.dtype
    target = args.target
    res = args.res
    tvm_file = args.file
    model_path = args.save_model_path
    if len(model_path)==0:
        model_path = tvm_file.replace('tvm', 'mlflow').replace('.so','.mlflow')


    # save model
    artifacts = {"tvm_executable": tvm_file}

    PYTHON_VERSION = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    conda_env = {
        "channels": ["defaults"],
        "dependencies": [
            f"python={PYTHON_VERSION}",
            "pip",
            {
                "pip": [
                    f"mlflow=={mlflow.__version__}",
                    f"tvm=={tvm.__version__}",
                    f"numpy=={np.__version__}",
                ],
            },
        ],
        "name": "tvm_env",
    }

    mlflow.pyfunc.save_model(path=model_path,
                             python_model=TVMWrapper(),
                             artifacts=artifacts,
                             conda_env=conda_env
                             )

    # load model
    loaded_model = mlflow.pyfunc.load_model(model_path)

    model_input = np.random.uniform(size=(1,3,res,res)).astype(dtype)
    model_output = loaded_model.predict(model_input)
    print(model_output)


