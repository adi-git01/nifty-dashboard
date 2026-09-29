#!/usr/bin/env bash
# Commit the given paths and push to the current branch WITHOUT discarding
# anything pushed in the meantime: rebase onto the remote, then push, retrying.
#
# Replaces git-auto-commit-action with push_options --force-with-lease. That
# action fetches right before pushing, so the lease always matched the latest
# remote and the force push overwrote whatever had landed during the run. On
# 29 Sep 2026 an engine run wiped a results commit pushed five minutes
# earlier; a PR merged mid-run would have been reverted the same way.
#
# Usage: scripts/commit_and_push.sh "commit message" path-or-glob [...]
# On a rebase conflict in the same file, this run's version wins (-X theirs:
# during a rebase, "theirs" is the commit being replayed -- this run's data).
set -uo pipefail
msg="$1"; shift
branch="${GITHUB_REF_NAME:-$(git rev-parse --abbrev-ref HEAD)}"
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
[ -f "$(git rev-parse --git-dir)/shallow" ] && git fetch -q --unshallow origin || true
for p in "$@"; do git add -- "$p" 2>/dev/null || true; done
if git diff --cached --quiet; then echo "Nothing to commit"; exit 0; fi
git commit -q -m "$msg"
for i in 1 2 3 4 5; do
  if git pull -q --rebase -X theirs origin "$branch" && git push -q origin "HEAD:$branch"; then
    echo "Pushed to $branch (attempt $i)"; exit 0
  fi
  git rebase --abort 2>/dev/null || true
  sleep $((i * 5))
done
echo "::error::Could not push after 5 attempts"; exit 1
