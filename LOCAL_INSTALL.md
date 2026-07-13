# Intel-Mac (x86_64) Setup Notes

`uv sync --dev` failed on an Intel Mac (`macosx_x86_64`) because several native
dependencies dropped Intel-Mac wheels. This repo assumes arm64 Mac / Linux.
Three separate issues surfaced, fixed in order below.

## 1. onnxruntime — no Intel-Mac wheel

**Error**
```
Distribution `onnxruntime==1.26.0` can't be installed because it doesn't have a
source distribution or wheel for the current platform (macosx x86_64).
```

**Chain**: `implementations/pyproject.toml` enables `aieng-forecasting[...,documents]`
→ `pymupdf4llm` → `pymupdf-layout` → `onnxruntime`.

**Root cause**: `onnxruntime==1.26.0` only ships wheels for arm64 Mac, Linux,
and Windows. onnxruntime's last Intel-Mac wheel is **1.23.2**. `pymupdf-layout`
does not pin onnxruntime, so an older, compatible version exists.

**Fix**: added `required-environments` to root `[tool.uv]` so uv must produce a
resolution valid on Intel Mac, forcing onnxruntime down to **1.23.2**.

```toml
[tool.uv]
required-environments = [
    "sys_platform == 'darwin' and platform_machine == 'x86_64'",
]
```

## 2. numba / llvmlite / numpy — no Intel-Mac wheel + numpy ceiling

**Error**
```
subprocess.CalledProcessError: Command '[.../llvmlite/0.47.0/.../build.py]'
returned non-zero exit status 1.
```
`llvmlite` has an sdist (unlike onnxruntime), so `required-environments` alone
let uv pick the newest and try to build it from source, which failed.

**Chain**: `aieng-forecasting[numerical]` → `statsforecast` → `numba` → `llvmlite`.

**Root cause (llvmlite build)**: newest `numba 0.65.1` / `llvmlite 0.47.0` have
no Intel-Mac wheels. Last Intel-Mac wheels: **numba 0.62.1**, **llvmlite 0.45.1**.
`statsforecast 2.0.3` allows `numba>=0.55.0`, so the older version is valid, and
numba 0.62.1 auto-selects llvmlite 0.45.1.

**Root cause (numpy ceiling)**: `numba 0.62.1` has a hard ceiling of
**numpy ≤ 2.3** (`numba._ensure_critical_deps()` raises `ImportError` if numpy
2.4+ is installed). Without an explicit numpy constraint the resolver picks numpy
2.4.x, breaking the `numba` import and therefore `statsforecast` and
`DartsAutoARIMAPredictor`.

**Fix**: two platform-scoped constraints — numba is capped only on Intel Mac;
numpy <2.4 is applied on all platforms because numba's ceiling is unconditional.

```toml
[tool.uv]
constraint-dependencies = [
    "numba<=0.62.1; sys_platform == 'darwin' and platform_machine == 'x86_64'",
    "numpy<2.4",
]
```

## 3. LightGBM — missing libomp (runtime)

**Error**
```
OSError: dlopen(.../lib_lightgbm.dylib): Library not loaded: @rpath/libomp.dylib
```

**Root cause**: not a packaging issue. The LightGBM wheel dynamically links
OpenMP (`libomp.dylib`), a system library that must be installed separately.

**Fix**:
```bash
brew install libomp
```
libomp is keg-only but lands at `/usr/local/opt/libomp/lib/libomp.dylib`, which
is one of the paths LightGBM's loader searches.

## Result

`uv lock && uv sync --dev` succeeds. Verified imports on Intel Mac:
onnxruntime 1.23.2, numba 0.62.1, llvmlite 0.45.1, numpy 2.3.5, statsforecast,
lightgbm 4.6.0, `DartsAutoARIMAPredictor`.

## Notes

- The `numba` constraint in `constraint-dependencies` is **platform-scoped** (Intel
  Mac only). The `numpy<2.4` constraint is not platform-scoped because numba's
  numpy ceiling applies on every platform — however, because numba 0.63+ supports
  numpy 2.4+, this only bites users who are pinned to an older numba (Intel Mac).
- The `VIRTUAL_ENV ... does not match the project environment path .venv`
  warning is harmless — it's the pyenv `vector` env vs. uv's `.venv`. Silence it
  by deactivating the pyenv env or running uv with `--active`.

# Corporate proxy / custom CA (TLS interception)

On a corporate laptop (e.g. Roche), data downloads can fail with an SSL error
even though the target site is public and healthy.

**Error** (seen in a VSCode Jupyter notebook downloading a StatCan table)
```
SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED]
certificate verify failed: self-signed certificate in certificate chain'))
host='www150.statcan.gc.ca'
```

**Root cause**: the corporate proxy (Cloudflare Gateway / Cisco Umbrella)
intercepts HTTPS and re-signs it with an internal CA. Python's `requests`
(used by `stats-can`, `fredapi`, etc.) trusts only its bundled `certifi` store,
which does **not** include the corporate CA — so verification fails. The macOS
Keychain *does* trust the corporate CA (IT installs it), which is why browsers
and `curl` work. Interception is often **intermittent** (depends on VPN /
proxy-client state), so the same download can pass off-proxy and fail on it.

**Fix**:
```bash
uv add pip-system-certs
```
`pip-system-certs` drops a `.pth` that runs at interpreter startup and redirects
Python's TLS trust to the **OS trust store** (macOS Keychain), covering
`requests` / `httpx` / `urllib`. Because it is a declared, locked dependency it
survives `uv run` syncs and applies to **both** the Jupyter kernel and CLI
scripts (`uv run python scripts/fetch_*.py`). After adding it, **restart the
Jupyter kernel** so the new interpreter picks it up.

**Cross-platform safety**: on Linux / CI the OS trust store is the system
`ca-certificates` (public roots only), so behavior is unchanged there — the
package only starts mattering on machines that have an extra corporate CA
installed.

**Alternative** (no dependency): export the corporate CA and point Python at a
*combined* bundle (public roots **must** stay included, or non-intercepted
public sites break):
```bash
cat "$(.venv/bin/python -m certifi)" ~/corp-ca.pem > ~/.config/combined-ca.pem
export SSL_CERT_FILE=~/.config/combined-ca.pem
export REQUESTS_CA_BUNDLE=~/.config/combined-ca.pem
export CURL_CA_BUNDLE=~/.config/combined-ca.pem
```
This is static and must be regenerated when `certifi` or the corporate CA
rotates; the `pip-system-certs` / Keychain approach auto-rotates.

# SSL errors in VS Code Jupyter notebooks (`curl_cffi` / yfinance)

**Error** (seen when the SP500 notebook fetches price data from Yahoo Finance):
```
CertificateVerifyError: Failed to perform, curl: (60) SSL certificate problem:
self signed certificate in certificate chain.
```

**Why `pip-system-certs` does not help here**: `pip-system-certs` patches
Python's `ssl` module, `requests`, and `httpx` to trust the macOS Keychain.
`yfinance` uses `curl_cffi` as its HTTP backend — a C extension that bundles
its own `libcurl`. Because `libcurl` is native code, it has its own SSL stack
and completely ignores Python's patching. The corporate proxy's CA is not in
`libcurl`'s default trust store, so every `yfinance` request fails when the
proxy intercepts it.

**Fix — automated (already in the notebook setup cell)**:

The setup cell in `01_sp500_multivariate_backtest.ipynb` runs the following
at kernel start, before any `yfinance` call:

```python
import certifi, subprocess, os
from pathlib import Path

combined = Path.home() / ".cache" / "agentic-forecasting" / "combined-ca.pem"
combined.parent.mkdir(parents=True, exist_ok=True)
kc = subprocess.run(
    ["security", "find-certificate", "-a", "-p", "/Library/Keychains/System.keychain"],
    capture_output=True, text=True,
)
combined.write_text(Path(certifi.where()).read_text() + "\n" + kc.stdout)
os.environ["CURL_CA_BUNDLE"] = str(combined)       # libcurl (curl_cffi)
os.environ["SSL_CERT_FILE"] = str(combined)        # Python ssl module
os.environ["REQUESTS_CA_BUNDLE"] = str(combined)  # requests / httpx
```

It exports every certificate from the macOS System Keychain, appends them to
certifi's public root bundle, writes the result to
`~/.cache/agentic-forecasting/combined-ca.pem`, and sets `CURL_CA_BUNDLE` so
that `libcurl` trusts both the corporate CA and all public CAs. It runs only
when `CURL_CA_BUNDLE` is not already set, so it's a no-op if you configure it
externally.

**Fix — manual (for scripts / CLI outside a notebook)**:

```bash
mkdir -p ~/.cache/agentic-forecasting
security find-certificate -a -p /Library/Keychains/System.keychain > /tmp/kc.pem
cat "$(.venv/bin/python -m certifi)" /tmp/kc.pem \
  > ~/.cache/agentic-forecasting/combined-ca.pem

export CURL_CA_BUNDLE=~/.cache/agentic-forecasting/combined-ca.pem
export SSL_CERT_FILE=~/.cache/agentic-forecasting/combined-ca.pem
export REQUESTS_CA_BUNDLE=~/.cache/agentic-forecasting/combined-ca.pem
```

Add those three `export` lines to your shell profile (`~/.zshrc`) to make
them permanent, or re-run them each shell session before `uv run`.

**Regeneration**: the combined bundle is static — regenerate it when `certifi`
is updated (`uv sync`) or when IT rotates the corporate CA certificate.
