🚀 Installation
1. Clone the repository (with submodules)
```bash
git clone --recurse-submodules https://github.com/YOUR_USERNAME/baba-gymnasium-env.git
cd baba-gymnasium-env
```


If you forgot --recurse-submodules:

```bash
git submodule update --init --recursive
```



Your directory structure should now look like:

```bash
baba-gymnasium-env/
    babelib/            # <-- full Baba-Is-Auto engine (submodule)
    my_baba_env/        # <-- the Gymnasium environment
    scripts/
    solvers/            # <-- PPO algorithm code and scripts to run it
    more_maps/          # <-- additional maps for demonstrating results
    README.md
```

2. Build the Baba engine (pyBaba)

```bash
cd babelib
mkdir build
cd build
cmake ..
make -j
```

This produces:

babelib/build/pyBaba.so

3. Install the Gymnasium environment

From the repo root:
```bash
pip install -e my_baba_env
```


If this does not work, install Python bindings:
From the root of the engine repo (babelib/):

```bash
pip install -U .
```

This installs the correct pyBaba module.

4. Run baba_simple_script.py

Navigate to baba-gymnasium-env/solvers and run:

```bash
python3 baba_simple_script.py
```

Follow the steps given in the terminal to run the PPO algorithm on one of six maps, with user chosen learning rate, reward discount, and clip bounds.

