# References

This directory stores non-normative source material used to design the harness.

The root `AGENTS.md` distills the Huawei-side Ascend operator engineering guidance supplied for this project into competition-safe rules: freeze the platform contract, keep implementation decisions in Host Tiling / tiling data / workspace, validate before performance claims, and avoid visible-case hardcoding.

External mutable repositories are not vendored here. `scripts/bootstrap_skills.sh` clones them into ignored `.agent-deps/`.
