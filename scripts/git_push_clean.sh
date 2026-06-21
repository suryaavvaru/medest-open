#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 scripts/check_repo_clean.py || true

git status

git add README.md .gitignore docs results examples paper scripts medest/training requirements.txt pyproject.toml
if git diff --cached --quiet; then
  echo "[OK] nothing staged; repo already clean"
else
  git commit -m "Upgrade MedEST-Open research scaffold"
fi

git branch -M main
if ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin https://github.com/suryaavvaru/medest-open.git
fi

git push -u origin main
