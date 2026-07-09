## 安装配置

## 1. 创建虚拟环境

建议在虚拟环境中运行训练或部署程序，推荐使用 Conda 创建虚拟环境。

### 1.1 创建新环境

使用以下命令创建虚拟环境：

```bash
conda create -n g1_deploy python=3.8
```

### 1.2 激活虚拟环境

```bash
conda activate g1_deploy
```

---

## 2. 安装依赖

### 2.1 安装 PyTorch

#### 2.1.1 x86_64 架构

```
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

#### 2.1.2 ARM64 架构

```bash
export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
# 针对俄罗斯使用 NVIDIA 官方全球/美国站点
# export TORCH_INSTALL=https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
python3 -m pip install --upgrade pip
python3 -m pip install --no-cache $TORCH_INSTALL
```

### 2.2 安装 G1_Deploy

#### 2.2.1 下载

通过 Git 克隆仓库：

```bash
git clone https://github.com/yuyyds/G1_deploy.git
```

#### 2.2.2 安装组件

进入目录并安装：

```bash
cd G1_deploy
pip install numpy==1.20.0
pip install onnx onnxruntime pydantic PyYAML scipy
```
#### 2.2.3 安装 unitree_sdk2_python

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```

**若遇到报错：**

```bash
Collecting cyclonedds==0.10.2 (from unitree_sdk2py==1.0.1)
  Downloading cyclonedds-0.10.2.tar.gz (156 kB)
  Installing build dependencies ... done
  Getting requirements to build wheel ... error
  error: subprocess-exited-with-error
  
  × Getting requirements to build wheel did not run successfully.
  │ exit code: 1
  ╰─> [1 lines of output]
      Could not locate cyclonedds. Try to set CYCLONEDDS_HOME or CMAKE_PREFIX_PATH
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: subprocess-exited-with-error

× Getting requirements to build wheel did not run successfully.
│ exit code: 1
╰─> See above for output.

note: This error originates from a subprocess, and is likely not a problem with pip.
```

需要先安装底层的 **C/C++ cyclonedds** 原生库：

1. **安装编译工具：**

   ```bash
   sudo apt update
   sudo apt install cmake build-essential
   ```

2.  **下载并安装 CycloneDDS C 库：**

   ```bash
   cd ~
   git clone -b 0.10.2 https://github.com/eclipse-cyclonedds/cyclonedds.git
   
   cd cyclonedds
   mkdir build install
   cd build
   
   # 配置 CMake (指定安装路径到 ../install 方便管理，或者安装到系统默认路径)
   cmake .. -DCMAKE_INSTALL_PREFIX=../install
   
   # 编译并安装
   cmake --build . --target install
   ```

3. **设置环境变量**

   ```bash
   # 注意：这里的路径要换成你刚才 cyclonedds/install 的绝对路径
   export CYCLONEDDS_HOME=~/cyclonedds/install
   ```

4. **再次尝试安装 Unitree SDK：**

```bash
cd ~/G1_deploy/unitree_sdk2_python
pip install -e .
```

#### 2.2.4 安装 unitree_sdk2

**在构建或运行 SDK 之前，确保安装了以下依赖：**

- CMake (version 3.10 or higher)
- GCC (version 9.4.0)
- Make

**安装所需的软件包：**

```bash
sudo apt-get update
sudo apt-get install -y cmake g++ build-essential libyaml-cpp-dev libeigen3-dev libboost-all-dev libspdlog-dev libfmt-dev
```

**克隆仓库并编译：**

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2.git
mkdir build
cd build
cmake ..
make
```

**若已经编译想要重新编译：**

```bash
cd unitree_sdk2
rm -rf build
mkdir build && cd build
cmake ..
make
```

#### 2.2.5 安装 unitree_cpp

```bash
cd unitree_cpp
pip install .
```

若遇到报错：

```bash
*** Building project with Ninja...
      [1/5] Building CXX object CMakeFiles/debugcpp.dir/src/unitree_controller.cpp.o
      FAILED: [code=1] CMakeFiles/debugcpp.dir/src/unitree_controller.cpp.o
      /usr/bin/g++  -pthread -B /home/unitree/miniconda3/envs/g1_deploy/compiler_compat -Wl,--sysroot=/   -I/usr/local/include/ddscxx -O3 -DNDEBUG   -g -O0 -std=gnu++17 -MD -MT CMakeFiles/debugcpp.dir/src/unitree_controller.cpp.o -MF CMakeFiles/debugcpp.dir/src/unitree_controller.cpp.o.d -o CMakeFiles/debugcpp.dir/src/unitree_controller.cpp.o -c /home/unitree/G1_deploy/unitree_cpp/src/unitree_controller.cpp
      In file included from /home/unitree/G1_deploy/unitree_cpp/src/unitree_controller.cpp:1:
      /home/unitree/G1_deploy/unitree_cpp/src/unitree_controller.hpp:20:10: fatal error: unitree/idl/hg/IMUState_.hpp: No such file or directory
         20 | #include <unitree/idl/hg/IMUState_.hpp>
            |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      compilation terminated.
      [2/5] Building CXX object CMakeFiles/unitree_cpp.dir/src/unitree_controller.cpp.o
      FAILED: [code=1] CMakeFiles/unitree_cpp.dir/src/unitree_controller.cpp.o
      /usr/bin/g++  -pthread -B /home/unitree/miniconda3/envs/g1_deploy/compiler_compat -Wl,--sysroot=/  -DVERSION_INFO=1.0.3 -Dunitree_cpp_EXPORTS -I/usr/local/include/ddscxx -isystem /home/unitree/miniconda3/envs/g1_deploy/include/python3.8 -isystem /tmp/pip-build-env-lkf90vfy/overlay/lib/python3.8/site-packages/pybind11/include -O3 -DNDEBUG -fPIC   -std=gnu++17 -MD -MT CMakeFiles/unitree_cpp.dir/src/unitree_controller.cpp.o -MF CMakeFiles/unitree_cpp.dir/src/unitree_controller.cpp.o.d -o CMakeFiles/unitree_cpp.dir/src/unitree_controller.cpp.o -c /home/unitree/G1_deploy/unitree_cpp/src/unitree_controller.cpp
      In file included from /home/unitree/G1_deploy/unitree_cpp/src/unitree_controller.cpp:1:
      /home/unitree/G1_deploy/unitree_cpp/src/unitree_controller.hpp:20:10: fatal error: unitree/idl/hg/IMUState_.hpp: No such file or directory
         20 | #include <unitree/idl/hg/IMUState_.hpp>
            |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      compilation terminated.
      [3/5] Building CXX object CMakeFiles/unitree_cpp.dir/src/py_binding.cpp.o
      FAILED: [code=1] CMakeFiles/unitree_cpp.dir/src/py_binding.cpp.o
      /usr/bin/g++  -pthread -B /home/unitree/miniconda3/envs/g1_deploy/compiler_compat -Wl,--sysroot=/  -DVERSION_INFO=1.0.3 -Dunitree_cpp_EXPORTS -I/usr/local/include/ddscxx -isystem /home/unitree/miniconda3/envs/g1_deploy/include/python3.8 -isystem /tmp/pip-build-env-lkf90vfy/overlay/lib/python3.8/site-packages/pybind11/include -O3 -DNDEBUG -fPIC   -std=gnu++17 -MD -MT CMakeFiles/unitree_cpp.dir/src/py_binding.cpp.o -MF CMakeFiles/unitree_cpp.dir/src/py_binding.cpp.o.d -o CMakeFiles/unitree_cpp.dir/src/py_binding.cpp.o -c /home/unitree/G1_deploy/unitree_cpp/src/py_binding.cpp
      In file included from /home/unitree/G1_deploy/unitree_cpp/src/py_binding.cpp:5:
      /home/unitree/G1_deploy/unitree_cpp/src/unitree_controller.hpp:20:10: fatal error: unitree/idl/hg/IMUState_.hpp: No such file or directory
         20 | #include <unitree/idl/hg/IMUState_.hpp>
            |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      compilation terminated.
      ninja: build stopped: subcommand failed.
      
      *** CMake build failed
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for unitree_cpp
Failed to build unitree_cpp
ERROR: Failed to build installable wheels for some pyproject.toml based projects (unitree_cpp)
```

**原因**：编译器在系统头文件路径里找不到 `unitree/idl/hg/IMUState_.hpp`，而这个文件是 unitree_sdk2 的 IDL 生成头文件，属于 `libunitree_sdk2` 开发包的一部分

先确认文件到底在不在：

```bash
sudo find /usr/local /opt -name IMUState_.hpp 2>/dev/null
```

若没有，重新编译安装 unitree_sdk2（带 IDL）：

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DBUILD_IDL=ON ..     # 关键：生成并安装 IDL 头文件
make -j$(nproc)
sudo make install
sudo ldconfig
```

装完再查一次：

```bash
ls /usr/local/include/unitree/idl/hg/IMUState_.hpp
```

重新安装：

注意：scikit-build-core >=0.8 需修改 pyproject.toml，将 cmake.minimum-version 改为 cmake.version = ">=3.15" 才能正常编译安装。
```bash
cd ~/G1_deploy/unitree_cpp
pip install .
```

---
## 3. 运行代码

## 1. 运行Mujoco仿真代码
```bash
python deploy_mujoco/deploy_mujoco.py
```
## 2. 真机操作说明
```bash
python deploy_real/deploy_real.py
```
---

