🚀 Installation
1️⃣ Clone This Repository (with engine submodule)
git clone --recurse-submodules https://github.com/YOUR_USERNAME/baba-gymnasium-env.git
cd baba-gymnasium-env


If you forgot --recurse-submodules:

git submodule update --init --recursive


Your directory structure should now look like:

baba-gymnasium-env/
    baba_engine/        <-- full Baba-Is-Auto engine (submodule)
    my_baba_env/        <-- your Gymnasium environment
    scripts/
    README.md

2️⃣ Build & Install the Full Baba-Is-Auto Engine

The engine lives in:

baba_engine/

Build (Linux / macOS / WSL):
cd baba_engine
mkdir -p build
cd build
cmake ..
make -j$(nproc)

Install Python bindings:

From the root of the engine repo (baba_engine/):

pip install -U .


This installs the correct pyBaba module including:

Step() / Tick() turn processing

HOT / MELT / DEFEAT

Object interactions

Win / loss state resolution

🔥 This step is what fixes lava not killing Baba.
