#!/bin/zsh
# Copy the Natural Questions answer runs back from the VM (run folders, r1-r3
# shard links, job folders, build records) and re-point the shard links, which
# hold absolute VM paths, at the same folders on the Mac. Indexes and caches
# are not copied. Run from the repo root.
set -e
instabox ssh-config omarchy-mpv0 > ~/.ssh/instabox.conf
R=/data/workspace/RecRAG/runs/final500
L=runs/final500
for m in rag raptor graphrag hipporag; do
  rsync -a -e "ssh -F $HOME/.ssh/instabox.conf" \
    --include='2026*/***' --include='r[123]/***' --include='build/***' --exclude='*' \
    instabox-omarchy-mpv0:$R/natural_questions/$m/ $L/natural_questions/$m/
done
rsync -a -e "ssh -F $HOME/.ssh/instabox.conf" --include='nq_*/***' --include='nq_*.log' --include='vm_queue*' --include='vm_queue/***' --include='vm_mem.log' --exclude='*' \
  instabox-omarchy-mpv0:$R/jobs/ $L/jobs/
for link in $L/natural_questions/*/r[123]/shard*(@); do
  target=$(readlink $link)
  ln -sfn "$PWD/$L/natural_questions/${${target#$R/natural_questions/}}" $link
done
ls -l $L/natural_questions/*/r[123]/shard*(@) | awk '{print $NF}' | head -3
