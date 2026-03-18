#!/bin/bash
set -e

cd /home/core
git reset --hard
bash /home/check_git_changes.sh
git checkout 5a8aa0b2ba575e098cbb63b396e9bcb751eb3a0f
bash /home/check_git_changes.sh

pnpm install || true

