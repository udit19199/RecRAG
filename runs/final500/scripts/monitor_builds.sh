#!/bin/zsh
# Writes runs/final500/STATUS.md for the builds until all 8 build.json exist
# ("stage: done") or no build process is left with some missing ("stage: failed").
cd /Users/uditagarwal19199/Workspace/HPE/RecRAG
R=runs/final500
st() { printf "stage: %s\nprogress: %s\nlast error: %s\nupdated: %s\nspend so far: %s\n" "$1" "$2" "$3" "$(date -u +%FT%TZ)" "$(.venv/bin/python $R/scripts/spend.py 2>/dev/null)" > $R/STATUS.md; }
while :; do
  n=$(ls $R/*/*/build/build.json 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" = 8 ]; then st done "8 of 8 builds done" "none"; exit 0; fi
  if ! pgrep -f "final500/scripts/build.py" >/dev/null; then
    missing=""; for f in $R/{hotpotqa,natural_questions}/{rag,raptor,hipporag,graphrag}; do [ -f $f/build/build.json ] || missing="$missing ${f#$R/}"; done
    st failed "$n of 8 builds done" "no build running; missing:$missing"; exit 1
  fi
  st "running (builds)" "$n of 8 builds done; running: $(pgrep -fl 'final500/scripts/build.py' | awk '{print $4"/"$5}' | tr '\n' ' ')" "see build/build_errors.log"
  sleep 120
done
