#!/bin/bash

conda uninstall -y mlflow
git clone https://github.com/mlflow/mlflow.git submodules/mlflow
cd submodules/mlflow
git remote add dbczumar https://github.com/dbczumar/mlflow.git
git fetch dbczumar
git checkout materialize_metrics
pip install -e .

