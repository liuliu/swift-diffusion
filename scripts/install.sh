#!/usr/bin/env bash

set -euo pipefail

GIT_CONFIG=$(git rev-parse --git-dir)
GIT_ROOT=$(git rev-parse --show-toplevel)

mkdir -p $GIT_CONFIG/hooks/pre-commit.d

rm -f $GIT_CONFIG/hooks/pre-commit
ln -s $GIT_ROOT/scripts/vendors/dispatch $GIT_CONFIG/hooks/pre-commit

rm -f $GIT_CONFIG/hooks/pre-commit.d/swift-format
ln -s $GIT_ROOT/scripts/swift-format/pre-commit $GIT_CONFIG/hooks/pre-commit.d/swift-format

rm -f $GIT_CONFIG/hooks/pre-commit.d/buildifier
ln -s $GIT_ROOT/scripts/buildifier/pre-commit $GIT_CONFIG/hooks/pre-commit.d/buildifier
