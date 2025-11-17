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

2. Build the Baba engine (pyBaba)
```bash
cd babelib
python3 setup.py build
python3 setup.py install --user
```

This produces:

babelib/build/pyBaba.so

3. Install the Gymnasium environment

From the repo root:
```bash
pip install -e my_baba_env
```
