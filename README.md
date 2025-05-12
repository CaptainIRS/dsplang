# DSPLang

> [!WARNING] 
> WIP: This is a work in progress.

## Overview

DSPLang is a programming language designed for building efficient vector operations targeting architectures with very wide registers. In particular, it is designed to be used with the Hexagon Vector eXtensions (HVX) of the Qualcomm Hexagon platform.

Details about the current implementation can be found [here](./DSPLang.pdf).

![image](https://github.com/user-attachments/assets/236f87cb-ad9f-445a-a541-ee8e3b638b9c)

### New dialects

* HLHVX dialect: A high-level dialect with the correct type information, suitable as a lowering target from other dialects in the MLIR ecosystem.
* LLHVX dialect: A one-to-one mapping to the HVX LLVM intrinsics, which directly translate to LLVM IR.

## Setup

Clone the repository.

### Setting up LLVM for the Hexagon platform

1. Using conda is recommended. `conda env create -f setup/hexagon-env.yaml`, 
   `conda activate hexagon`
2. `git clone https://github.com/quic/toolchain_for_hexagon`
3. `cd toolchain_for_hexagon`
4. `git apply ../setup/hexagon-toolchain.patch`
5. `./build.sh`

### Building DSPLang

1. Create a python environment: `python3 -m venv venv`
2. Activate the environment: `source venv/bin/activate`
3. Install the required packages: `pip install -r requirements.txt`
2. `mkdir build`
3. `cd build`
4. `cmake -G Ninja -DMLIR_DIR=$PWD/toolchain_for_hexagon/clang+llvm-fork-cross-hexagon-unknown-linux-musl/x86_64-linux-gnu/lib/cmake/mlir -DCMAKE_PREFIX_PATH=$PWD/venv/bin -DLLVM_EXTERNAL_LIT=$PWD/venv/bin/lit ..`
5. `ninja`

### Running the examples (simulator)

Note: You need to have the Hexagon SDK installed as the examples use functionalities from the SDK for instrumentation. Make sure that the environment variables are set up correctly.

1. `cd examples`
2. `make sim EXP=basic EXP=dsp` to run the DSPLang version of the `basic` example.

### Running the examples (device)

Note: Only the QCS8550 SoC (Snapdragon 8 Gen 2) is supported at the moment. Connect the device to your computer and make sure that `adb devices -l` shows the device.

1. `cd examples`
2. `make device EXP=basic EXP=dsp` to run the DSPLang version of the `basic` example.

