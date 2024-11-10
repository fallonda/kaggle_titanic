#!/usr/bin/env bash

set -euxo pipefail

python -m venv .venv

source .venv/bin/activate

pip install --upgrade pip

pip install -r ./requirements.txt

