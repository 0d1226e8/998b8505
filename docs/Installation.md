# Installation

* [Dependencies](#dependencies)
  - [Linux](#dependencies_linux)
* [Installation](#installation)
  - [via PyPI](#installation_pypi)
  - [via Docker](#installation_docker)

## <a name="dependencies"></a> Dependencies

There are a few system dependencies for BDL Benchmarks:

* [Python 3](https://www.python.org/) (>=3.5)
* [p7zip](http://p7zip.sourceforge.net/)
* [wget](https://www.gnu.org/software/wget/)
* [CUDA](https://www.nvidia.co.uk/)

### <a name="dependencies_linux"></a> Linux

For Ubuntu 18.04, `python3` is already installed:

```bash
# Update & upgrade packages
sudo apt-get update && sudo apt-get -y upgrade

# Install python3 PyPI and setuptools
sudo apt-get install python3-pip python3-setuptools

# Installs p7zip
sudo apt-get install p7zip-full

# Installs wget
sudo apt-get install wget

# Install CUDA 10
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda-drivers
sudo reboot
sudo apt-get update && sudo apt-get -y upgrade
rm cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
echo 'export CUDA_HOME=/usr/local/cuda' | sudo tee -a /etc/bash.bashrc
echo 'export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}' | sudo tee -a /etc/bash.bashrc
echo 'export PATH=${CUDA_HOME}/bin:${PATH}' | sudo tee -a /etc/bash.bashrc
source /etc/bash.bashrc
```

### <a name="dependencies_macos"></a> macOS

For macOS Mojave or higher [homebrew](https://brew.sh/) can be used for system dependencies:

```bash
# Installs Python 3.7
brew install python

# Installs p7zip
brew install p7zip

# Installs wget
brew install wget
```

## <a name="installation"></a> Installation

BDL Benchmarks is a Python package that can by install with `setuptools`.

### <a name="installation_pypi"></a> Installation via PyPI (recommended for Python users)

BDL Benchmarks for Python can be installed via **pip/conda** on Linux and macOS, and it is strongly recommended.
However you will still need to install **[Linux](#dependencies_linux)/[macOS](#dependencies_macos) dependencies**.

To install the most stable official release from [PyPI](https://pypi.org/):

```bash
pip3 install bdl-benchmarks
```

To install the newest version from the repository:

```bash
git clone https://github.com/oatml/bdl-benchmarks
cd bdl-benchmarks
pip3 install -e .
```

Or without cloning it yourself:

```bash
pip3 install git+https://github.com/oatml/bdl-benchmarks
```

### <a name="installation_docker"></a> Installation via Docker

The [Dockerfile](../docker/min.Dockerfile) can be also used for setting up the system and Python dependencies.