#!/usr/bin/env bash
# Token-pair softmax weight dumps for the multi-target contact validation
# (R2 Q6).  One run per shot at sample 1; after the normal screening pass a
# dedicated pass saves the TMI weights of ~300 query molecules per target
# into tmi_weights_dump/ (consumed by script/analyze_contact_validation.py).
# 4 DUD-E runs.
source "$(dirname "$0")/_ab_common.sh"

SAMPLES=(1)
run_matrix tmidump \
  --dump-tmi-weights "${ROOT}/tmi_weights_dump"
