#!/bin/bash
# Run a command against the PRIVATE patched Honeykrisp driver (got-bringup RPM,
# extracted, never installed). Only this process and its children see it; the
# system driver in /usr/lib64 is untouched. All deps resolve against system
# libs (ldd checked), so no LD_LIBRARY_PATH is needed.
ICD="$(dirname "$(readlink -f "$0")")/patched_icd.json"
export VK_DRIVER_FILES="$ICD"
export VK_ICD_FILENAMES="$ICD"
exec "$@"
