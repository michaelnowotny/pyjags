# Installation

## Option 1: Docker (fastest, any platform)

If you have Docker installed, PyJAGS includes a ready-to-run environment
with JAGS, Python, and Jupyter Lab pre-configured:

```bash
git clone https://github.com/michaelnowotny/pyjags.git
cd pyjags
cp .env.example .env
./scripts/jagslab build   # build Docker image
./scripts/jagslab start   # launch Jupyter Lab at http://localhost:8888
```

## Option 2: Native (macOS)

### Apple Silicon (M1/M2/M3/M4)

```bash
# 1. Install JAGS
brew install jags

# 2. Set PKG_CONFIG_PATH (add to ~/.zprofile)
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"

# 3. Python environment with uv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate

# 4. Install PyJAGS
uv pip install pyjags

# 5. Optional: advanced diagnostics
uv pip install pyjags[diagnostics]
```

### Intel Mac

```bash
brew install jags
pip install pyjags
```

## Option 3: Native (Linux)

```bash
# Debian/Ubuntu
sudo apt-get install jags pkg-config
pip install pyjags
```

## Windows

Use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)
with Ubuntu, then follow the Linux instructions.

## From Source

```bash
git clone https://github.com/michaelnowotny/pyjags.git
cd pyjags
pip install -e ".[dev]"
pre-commit install
```

## Troubleshooting

### macOS: `symbol not found '_JAGS_NA'`

Verify `pkg-config --libs --cflags jags` succeeds. If not, set
`PKG_CONFIG_PATH` as shown above.

### macOS: architecture mismatch

Install JAGS via the ARM Homebrew at `/opt/homebrew`:

```bash
/opt/homebrew/bin/brew install jags
```

### CMake errors

```bash
brew install cmake    # macOS
sudo apt install cmake  # Linux
```
