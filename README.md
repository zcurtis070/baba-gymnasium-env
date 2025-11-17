🚀 Installation
1. Clone the repository (with submodules)
git clone --recurse-submodules https://github.com/YOUR_USERNAME/baba-gymnasium-env.git
cd baba-gymnasium-env


If you forgot --recurse-submodules:

git submodule update --init --recursive

⚙️ Build the Baba engine (pyBaba)
cd babelib
mkdir build
cd build
cmake ..
make -j


This produces:

babelib/build/pyBaba.so

📦 Install the Gymnasium environment

From the repo root:

cd my_baba_env
pip install -e .
