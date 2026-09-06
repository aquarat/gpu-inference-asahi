#!/bin/bash
VK=${VK:-$(dirname "$(readlink -f "$0")")}; FRIGATE_HOME=${FRIGATE_HOME:-$HOME/frigate}; IMMICH_WORK=${IMMICH_WORK:-$HOME/immich-ml/work}
# Private second detector instance on port 5556 (never the live :5555 service), stock then patched, 60 s at 25 req/s.
WP=$VK/with-patched.sh
R=$VK/results
DET=$FRIGATE_HOME/detector
PRIV=$VK/detector-models
T0=$(date -u +%FT%TZ); echo "start $T0"
for drv in stock patched; do
  if [ $drv = patched ]; then pre=$WP; else pre=env; fi
  echo "=== $drv instance"
  systemd-run --user --scope -q -p MemoryMax=3G $pre $DET/venv/bin/python $DET/detector/zmq_onnx_client.py \
      --backend ncnn-vulkan --threads 4 --endpoint tcp://127.0.0.1:5556 --models-dir $PRIV --model AUTO \
      > $R/detector_$drv.log 2>&1 &
  SPID=$!
  for i in $(seq 1 60); do ss -ltn | grep -q ':5556 ' && break; timeout 1 tail -f /dev/null; done
  ss -ltn | grep ':5556 ' || { echo "server did not bind"; cat $R/detector_$drv.log; kill $SPID; exit 1; }
  # warm-up: 20 requests (loads the model on the first handshake)
  $DET/venv/bin/python $DET/test_client.py --endpoint tcp://127.0.0.1:5556 --model-name fox-person-cat-320.onnx -n 20 --timeout-ms 5000 > $R/client_${drv}_warmup.log 2>&1
  $DET/venv/bin/python $DET/test_client.py --endpoint tcp://127.0.0.1:5556 --model-name fox-person-cat-320.onnx --duration 60 --rate 25 --report-every 20 --timeout-ms 1000 > $R/client_$drv.log 2>&1
  tail -4 $R/client_$drv.log
  pkill -TERM -f 'endpoint tcp://127.0.0.1:5556'; wait $SPID 2>/dev/null
  for i in $(seq 1 10); do ss -ltn | grep -q ':5556 ' || break; timeout 1 tail -f /dev/null; done
  grep -iE 'backend|driver|vulkan|ready|error|Traceback' $R/detector_$drv.log | head -12
done
echo "=== dmesg since $T0"; sudo dmesg --time-format iso | grep -iE 'agx|gpu|fault|timeout|asahi' | tail -20
echo "=== live service journal (last 30 min)"; journalctl -u frigate-detector --since -30min --no-pager | grep -viE 'inference|request' | tail -15
echo "=== live service state"; systemctl show frigate-detector -p ActiveEnterTimestamp -p NRestarts -p MainPID
echo DET_DONE
