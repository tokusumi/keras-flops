#!/usr/bin/env bash

set -e

pytest --cov=keras_flops --cov=tests --cov-report=xml --disable-warnings tests/