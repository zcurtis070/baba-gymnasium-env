📦 Installation
1. Clone the repository (with submodules)
git clone --recurse-submodules https://github.com/YOUR_USERNAME/baba-gymnasium-env.git
cd baba-gymnasium-env


If you accidentally cloned without --recurse-submodules:

git submodule update --init --recursive

⚙️ Build the Baba engine (pyBaba)

The engine source lives in:

babelib/


Build it:

cd babelib
mkdir build
cd build
cmake ..
make -j


This produces:

build/pyBaba.so


Gymnasium will load this automatically.

🏗️ Install the Gymnasium environment

From the repo root:

cd my_baba_env
pip install -e .


This installs:

my_baba_env/
  └── BabaIsYou-v1 (Gymnasium-compatible)
