# MhcExpand isolated CANN 8.5 environment

## Platform target

- CANN: `8.5.0`
- SOC: `ascend910b` / Atlas A2

## Host

- OS: openEuler 22.03 (LTS-SP4)
- Architecture: `aarch64`
- Device: Ascend910 (board query name `9382`)
- Driver: `25.5.0`, inner version `V100R001C23SPC005B219`
- Firmware: `npu-smi` reports `NA`
- Driver warning: `npu-smi` reports that the driver package may not be completely installed.

The warning is reproducible at the end of `npu-smi info`. The expected
`/etc/ascend_install.info` path is an empty directory on this host; the regular
file `/etc/ascend_install.info.bak` records `Driver_Install_Status=complete`.
This metadata path/type mismatch plausibly triggers the warning, but is not by
itself evidence that the loaded Driver modules or libraries are incomplete.

## System environment

- Existing CANN: `/usr/local/Ascend/cann-9.0.0-beta.1`
- An additional `/usr/local/Ascend/cann-9.0.0-beta.2` Toolkit tree exists.
- Neither system tree, the system `latest` link, Driver, nor Firmware was modified.

## Isolated environment

- Competition build root: `$HOME/ascend-envs/cann-8.5.0`
- Competition build CANN home: `$HOME/ascend-envs/cann-8.5.0/cann-8.5.0`
- Toolkit: stable `8.5.0`, inner version `V100R001C25SPC001B232`
- Competition build ops: `Ascend-cann-910b-ops` stable `8.5.0`, same inner version
- Environment script: `$HOME/ascend-envs/cann-8.5.0/cann-8.5.0/set_env.sh`
- Python venv: `$HOME/venvs/xingchenbei-cann85`
- Python: 3.9.9
- Python packages: NumPy 1.26.4 and pytest 8.3.5
- torch/torch-npu: not installed; direct ACL/ACLNN validation does not require them.

The competition build root remains the `ascend910b` / Atlas A2 environment.
The locally installed device reports runtime SOC `Ascend910_9382`, whose NNOP
lookup uses the `ascend910_93` directory. CANN does not support installing A2
and A3 ops packages into the same CANN home, so local A3 runtime validation has
a second, fully separate root:

- Local A3 runtime root: `$HOME/ascend-envs/cann-8.5.0-a3`
- Local A3 CANN home: `$HOME/ascend-envs/cann-8.5.0-a3/cann-8.5.0`
- Local A3 ops: `Ascend-cann-A3-ops` stable `8.5.0`, same inner version
- Local A3 wrapper: `tasks/mhc_expand/scripts/with-cann85-a3.sh`

The A3 root exists to make built-in ACLNN and locally compiled A3 binaries
runnable on this host. It does not change the competition submission SOC or
authorize changing the official template.

The official installer also writes user-level discovery metadata to
`$HOME/Ascend/ascend_cann_install.info`. It did not edit shell startup files.

## Official packages

Downloaded outside Git to `$HOME/ascend-packages/cann-8.5.0`:

- `Ascend-cann-toolkit_8.5.0_linux-aarch64.run`
  - SHA256: `bd702440a2b3bf1e0a07d321ed5cb55c181e954a82d0b8edd4b13039b37ef929`
- `Ascend-cann-910b-ops_8.5.0_linux-aarch64.run`
  - SHA256: `e7857be90f80c400b434e33fc5ad8ac477ec5684bf7c26fcb0aa839fb3742e32`
- `Ascend-cann-A3-ops_8.5.0_linux-aarch64.run`
  - SHA256: `d9440eaa4e733a9a0824de6c49e7a01e799a182c4e717ff78e8d581a561c5b6a`

The packages came from the official Huawei OBS URLs documented in the
[CANN 8.5.0 installation guide](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/softwareinst/instg/instg_0008.html).
Matching `.asc` files identify signing key `99AD81DF27A74824`, but that public
key is not installed in the local GPG keyring, so no local cryptographic GPG
verification is claimed. Every installed `.run` package passed its embedded
`--check` integrity verification before installation.

## Usage

Check isolation:

```bash
tasks/mhc_expand/scripts/with-cann85.sh \
  tasks/mhc_expand/scripts/check-cann85.sh
```

Check the separate local A3 runtime environment:

```bash
tasks/mhc_expand/scripts/with-cann85-a3.sh \
  tasks/mhc_expand/scripts/check-cann85.sh
```

Build the submission-target package. This fixed entry selects the platform
CANN 8.5 + 910B OPS wrapper itself and uses an independent clean build root:

```bash
tasks/mhc_expand/scripts/build-platform-910b.sh
```

Validate the same algorithm source on the local A3 device. This fixed entry
creates a temporary source mirror and changes only the two SOC target fields;
the resulting package is explicitly not for submission:

```bash
MHC_EXPAND_DEVICE_ID=4 tasks/mhc_expand/scripts/validate-local-a3.sh
```

`validate-local-a3.sh` defaults to device 4 when `MHC_EXPAND_DEVICE_ID` is not
set and rejects device 0. Set another device explicitly only after confirming
it is healthy and idle. The lower-level wrappers retain their general-purpose
device handling. The system CANN 9.0 environment can be invoked without
inheriting the caller environment using `scripts/with-cann90.sh`.

The platform entry owns `${TMPDIR:-/tmp}/mhc_expand_platform_910b`; the local
A3 entry owns `${TMPDIR:-/tmp}/mhc_expand_a3_validation` with separate
`source/`, `build/`, `install/`, and `logs/` directories. Each script cleans
only its exact managed root. They never share CMake cache or package output.

The wrapper launches `env -i` plus a non-login, non-interactive Bash child,
sources only CANN 8.5, validates exact version and path provenance, then uses
`exec` for the requested command. CANN 9 user-space paths cause immediate
failure. Shared `/usr/local/Ascend/driver` libraries remain available by design.
The A3 wrapper uses the same isolation mechanism and additionally requires
the installed ops metadata to name `Ascend-cann-A3-ops`.

## Evidence

- Environment and A/B isolation: [`runs/cann85-check.txt`](runs/cann85-check.txt)
- Clean build: [`runs/cann85-build.txt`](runs/cann85-build.txt)
- ACL runtime probe: [`runs/cann85-runtime.txt`](runs/cann85-runtime.txt)
- CANN 8.5/9.0 A/B and multi-device diagnosis:
  [`runs/acl-runtime-ab.txt`](runs/acl-runtime-ab.txt)
- Independent-process stability matrix for devices 2 and 4:
  [`runs/runtime-stability-device2-device4.txt`](runs/runtime-stability-device2-device4.txt)
- Staged CANN 8.5 device-4 correctness attempt and NNOP control:
  [`runs/cann85-correctness-device4.txt`](runs/cann85-correctness-device4.txt)
- CANN 8.5 NNOP lookup trace:
  [`runs/cann85-aclnn-abs-control-debug-plog.log`](runs/cann85-aclnn-abs-control-debug-plog.log)
- A3 ops post-install audit:
  [`runs/cann85-a3-post-install-audit.txt`](runs/cann85-a3-post-install-audit.txt)
- Post-fix built-in ACLNN full-lifecycle control:
  [`runs/cann85-a3-wrapper-aclnn-abs-control-pass.txt`](runs/cann85-a3-wrapper-aclnn-abs-control-pass.txt)
- Post-fix MhcExpand FP16 smoke and installed custom-package SOC audit:
  [`runs/cann85-a3-mhcexpand-fp16-smoke.txt`](runs/cann85-a3-mhcexpand-fp16-smoke.txt),
  [`runs/cann85-a3-mhcexpand-package-soc-audit.txt`](runs/cann85-a3-mhcexpand-package-soc-audit.txt)
- Dual-track platform build:
  [`runs/dual-track-platform-910b-build.txt`](runs/dual-track-platform-910b-build.txt)
- Dual-track A3 adaptation, build, ACLNN execution, and full correctness:
  [`runs/dual-track-local-a3-validation.txt`](runs/dual-track-local-a3-validation.txt)

The clean CANN 8.5 build passes without a 9.0-only source compatibility error.
Independent-process probes subsequently completed 5/5 CANN 8.5 and 5/5 CANN
9.0 lifecycles on device 2, plus 3/3 under each version on device 4. The selected
devices are therefore classified `RUNTIME STABLE`. A cold `aclrtSetDevice` can
take about 9–11 seconds; warm independent probes typically take 0.4–0.6 seconds.
This time is environment initialization, not operator latency. The separate
device-0 condition remains an administrator concern and is not used for
MhcExpand validation.

A current-state revalidation reproduced the same A/B result with a common
30-second timeout. A read-only `/proc` sample of a separate timeout-managed
device 0 probe showed three threads blocked in aarch64 syscall 29 (`ioctl`) on
fd 6, request `0x4816`; fd 6 resolved to `/dev/hisi_hdc`, and the kernel wait
channel was `hdcdrv_recv_peek_wait`. This is direct evidence of a
`DRIVER/DEVICE HDC IOCTL BLOCK` on device 0, not evidence of a CANN 8.5-only
userspace incompatibility. `strace` is not installed, so no package was added.

The original CANN 8.5 built-in `aclnnAbs` control failed with `561103`. The OP
debug trace proves NNOP requested `ascend910_93` metadata, while the original
isolated root contained only the installed `Ascend-cann-910b-ops` indexes and
Abs binaries under `ascend910b`. A clean shell and the repository wrapper had
the same core CANN variables and both reproduced the failure. The complete A3
ops installation in the separate root provides the required `ascend910_93`
indexes and binaries; the independent built-in control then passed
`GetWorkspaceSize`, executor creation, execution, synchronization, and exact
output comparison. Neither CANN 8.5 root has `opp/vendors/config.ini`, so that
file is not required for this built-in path and was not the root cause.

Only after the built-in control passed, one deterministic FP16 MhcExpand smoke
was run. The previously built custom package contains only `ascend910b`
binary-info and kernel artifacts, so the local A3 NNOP lookup failed with
`161001` before Host tiling or kernel launch because `ascend910_93` does not
index `MhcExpand`. No MhcExpand source, template, or CMake file was changed in
this investigation. This result proves a custom-package SOC coverage mismatch;
it is not a MhcExpand correctness result.

The subsequent dual-track workflow resolved that package-SOC mismatch without
changing the persistent competition target. The formal 910B entry clean-built
an `ascend910b` package. The local entry generated a temporary mirror whose
recursive diff contained only `ASCEND_COMPUTE_UNIT` and OpDef `AddConfig`, both
adapted to `ascend910_93`; its A3 package loaded and returned a valid executor.
All staged cases were exact, and the full suite passed 18/18. The correct label
for this evidence is `LOCAL A3 CORRECTNESS PASS`, not platform correctness.
