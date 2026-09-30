---
name: push-to-origin
description: Pushes this repo to GitHub so origin mirrors every local branch and step tag. Use whenever the user asks to push anything - a branch, main, a merge - because pushing only the named ref leaves the other branches on origin behind their merges.
---

# Pushing to origin

Origin mirrors the local repo: every branch and every `step-NN` tag, each at the same commit. Push only
when asked; when asked, push everything, not just the ref that was named. Pushing `main` after merging
step 04 without the step branch left `step/04-train-our-model` 13 commits stale on origin.

## Check what differs

```bash
git fetch -q origin
for b in $(git for-each-ref --format='%(refname:short)' refs/heads); do
    if git rev-parse -q --verify "origin/$b" > /dev/null; then
        echo "$b: ahead $(git rev-list --count "origin/$b..$b"), behind $(git rev-list --count "$b..origin/$b")"
    else
        echo "$b: not on origin"
    fi
done
git diff origin/main main -- third_party/pico-tflmicro    # must print nothing
```

- A branch **behind** origin means commits exist there that are not here. Stop and ask; never force-push.
- The submodule line must be empty: pico-tflmicro is patched at build time and its pointer is never
  committed (see the `pico-build-flash` skill).

## Push

```bash
git push -u origin --all     # every branch, with upstream tracking
git push origin --tags       # step-NN tags
```

Then run the check again: every branch should read `ahead 0, behind 0`.
