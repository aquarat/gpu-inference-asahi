#!/bin/bash
# Plain Fedora rawhide Mesa 26.2.2 Honeykrisp (no fork patches), extracted privately; control arm for attributing gains to upstream progress.
ICD="$(dirname "$(readlink -f "$0")")/upstream_icd.json"
export VK_DRIVER_FILES="$ICD" VK_ICD_FILENAMES="$ICD"
exec "$@"
