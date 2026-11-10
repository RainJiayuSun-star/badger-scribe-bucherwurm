#!/bin/bash
# Bring a local unpushed branch to an EFabric clone with git bundle over ssh.
# usage: send_branch.sh <branch> <clone name under home, e.g. badger-scribe-run-a>
# The clone then has the branch checked out; refuses if the clone has tracked changes.
set -euo pipefail
BRANCH=$1; CLONE=$2
REPO=/Users/iangloege/Desktop/badger-scribe-bucherwurm
HOST=igloege_ian-badgerscribe_0@128.105.218.242   # GPU 1; shares its home with GPU 2
SLUG=${BRANCH//\//-}-$(date +%Y%m%d%H%M%S)
TMP=$(mktemp -d)
git -C "$REPO" bundle create -q "$TMP/$SLUG.bundle" "$BRANCH" --not origin/ian/predict-script
ssh -p 2200 "$HOST" 'mkdir -p ~/pm-bundles && cat > ~/pm-bundles/'"$SLUG"'.bundle' < "$TMP/$SLUG.bundle"
ssh -p 2200 "$HOST" 'set -e; cd ~/'"$CLONE"'; test -z "$(git status --porcelain --untracked-files=no)"; git fetch -q ~/pm-bundles/'"$SLUG"'.bundle +'"$BRANCH"':'"$BRANCH"'; git checkout -q '"$BRANCH"'; git log -1 --format="%H %s"; git describe --always --dirty --abbrev=40'
