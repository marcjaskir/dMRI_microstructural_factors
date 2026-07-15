#!/usr/bin/env bash
# Create the private GitHub repo and push (run once you have gh or SSH/HTTPS auth).
set -euo pipefail
cd "$(dirname "$0")"

if command -v gh >/dev/null 2>&1; then
  gh repo create dMRI_microstructural_factors \
    --private \
    --source=. \
    --remote=origin \
    --description "Code for dMRI microstructural factor analysis and TLE asymmetry (publication companion)" \
    --push
  gh repo view --web
else
  echo "gh CLI not found. Create the private repo on GitHub, then:"
  echo "  git remote add origin git@github.com:marcjaskir/dMRI_microstructural_factors.git"
  echo "  git push -u origin main"
  echo
  echo "Or install gh (https://cli.github.com/) and re-run this script."
  exit 1
fi
