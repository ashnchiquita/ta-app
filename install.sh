#!/bin/bash

python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install resources/hailort-4.20.0-cp311-cp311-linux_aarch64.whl
