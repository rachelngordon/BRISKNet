#!/usr/bin/env bash
set -euo pipefail

qos="${1:-burst}"

sacctmgr show qos "${qos}" \
  format=Name,MaxWall,MaxJobsPU,MaxSubmitPU,MaxTRESPU%40,MaxTRESPerJob%40 \
  -n -P \
| awk -F'|' '{
    printf "Name: %s\n", $1
    printf "MaxWall: %s\n", $2
    printf "MaxJobsPU: %s\n", $3
    printf "MaxSubmitPU: %s\n", $4
    printf "MaxTRESPU: %s\n", $5
    printf "MaxTRESPerJob: %s\n", $6
  }'
