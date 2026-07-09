#!/usr/bin/env bash
# Fetch the upstream reference implementation (Hekimoglu & Gökgöz) for the C/Numba
# benchmark comparison. Not vendored here: the upstream repo carries no license,
# so we reference it by fetching at a pinned commit rather than redistributing.
#
#   Usage:  bash ref/fetch.sh
#
# Downloads into this ref/ directory. Only needed if you want to reproduce the
# C benchmark in RESULTS.md; the Julia solver itself has no dependency on these.
set -euo pipefail

REPO="variancegamma/A-Fast-IV-method-with-expansions"
SHA="e741041809ddc9fc6d797b94c1a35e375cbc7a70"   # pinned; upstream has no tags/license
BASE="https://raw.githubusercontent.com/${REPO}/${SHA}"
FILES=(iv_regime_final_v2.c bench_iv_c_all_hh4.c taylor_iv_efficiency_v6.py)

cd "$(dirname "$0")"
for f in "${FILES[@]}"; do
    echo "fetching $f @ ${SHA:0:12}"
    curl -fsSL "${BASE}/${f}" -o "$f"
done
echo "done. reproduce the C benchmark:"
echo "  cc -O3 -march=native -ffast-math bench_iv_c_all_hh4.c -o bench_c && ./bench_c"
