#!/bin/bash
VK=${VK:-$(dirname "$(readlink -f "$0")")}; FRIGATE_HOME=${FRIGATE_HOME:-$HOME/frigate}; IMMICH_WORK=${IMMICH_WORK:-$HOME/immich-ml/work}
# One arm of the private :5556 detector A/B. usage: run-detector-arm.sh stock|patched
drv=$1
WP=$VK/with-patched.sh
R=$VK/results
DET=$FRIGATE_HOME/detector
PRIV=$VK/detector-models
if [ $drv = patched ]; then pre=$WP; else pre=env; fi
T0=$(date -u +%FT%TZ); echo "start $T0 arm=$drv"
ss -ltn | grep -q ':5556 ' && { echo "port 5556 busy"; exit 1; }
systemd-run --user --scope -q -p MemoryMax=3G $pre $DET/venv/bin/python $DET/detector/zmq_onnx_client.py \
    --backend ncnn-vulkan --threads 4 --endpoint tcp://127.0.0.1:5556 --models-dir $PRIV --model AUTO \
    > $R/detector_$drv.log 2>&1 &
SPID=$!
for i in $(seq 1 60); do ss -ltn | grep -q ':5556 ' && break; timeout 1 tail -f /dev/null; done
ss -ltn | grep -q ':5556 ' || { echo "server did not bind"; cat $R/detector_$drv.log; kill $SPID; exit 1; }
$DET/venv/bin/python $DET/test_client.py --endpoint tcp://127.0.0.1:5556 --model-name fox-person-cat-320.onnx -n 20 --timeout-ms 5000 > $R/client_${drv}_warmup.log 2>&1
$DET/venv/bin/python $DET/test_client.py --endpoint tcp://127.0.0.1:5556 --model-name fox-person-cat-320.onnx --duration 60 --rate 25 --report-every 20 --timeout-ms 1000 > $R/client_$drv.log 2>&1
tail -3 $R/client_$drv.log
SRV=$(pgrep -f "zmq_onnx_client.py --backend ncnn-vulkan --threads 4 --endpoint tcp://127.0.0.1:555[6]" | head -1)
[ -n "$SRV" ] && kill -TERM $SRV; wait $SPID 2>/dev/null
for i in $(seq 1 10); do ss -ltn | grep -q ':5556 ' || break; timeout 1 tail -f /dev/null; done
grep -E 'ncnn-vulkan\]|Inference stats|Cleanup|Traceback|rror' $R/detector_$drv.log | head -8
echo "=== dmesg (gpu/agx) since $T0"; sudo dmesg --time-format iso | grep -iE 'agx|gpu|fault|timeout' | tail -10
echo "=== live service journal last 30 min (non-stats)"; journalctl -u frigate-detector --since -30min --no-pager | grep -viE 'Inference stats' | tail -8
echo "=== live service journal stats last 30 min"; journalctl -u frigate-detector --since -30min --no-pager | grep -E 'Inference stats' | tail -6
systemctl show frigate-detector -p ActiveEnterTimestamp -p NRestarts
echo ARM_DONE
