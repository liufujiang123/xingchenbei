# CANNJudge integration

The harness bootstraps the official CANN `cann-learning-hub` repository and exposes `cannjudge-submit` and its supporting `ascendc-ops-project` skill under `.agents/skills/`.

## What is automated

`tools/cannjudge_eval.py` reuses the official `cannjudge-submit/cannjudge_cli.py` client in-process. It can:

- verify the local skill/project wiring (`doctor`);
- log in with RSA ciphertext without storing or printing a plaintext password;
- submit the four CANNJudge source payloads from a task workspace;
- poll submission status;
- query a previous submission;
- inspect rankings;
- emit stable `CANNJUDGE_SUBMISSION_ID`, `CANNJUDGE_STATUS`, and `CANNJUDGE_SCORE` markers;
- save sanitized platform evidence under `tasks/<task>/runs/cannjudge/`.

`tools/agent_loop.py` now supports task-scoped configs and a `platform` mode. Local benchmark bests and CANNJudge bests are kept separately under `tasks/<task>/runs/harness/`.

## Safety rules

1. Never put a plaintext CANNJudge password in chat, config, shell history, or environment. `CANNJUDGE_PASSWORD` is explicitly rejected by the adapter.
2. Use the official RSA flow. Keep `private.pem` local with mode `0600`.
3. Keep ciphertext/key/session material out of Git.
4. Platform submission is an external side effect. `cannjudge_eval.py submit` requires `--yes-submit`; `agent_loop.py platform` separately requires `--submit`.
5. CANNJudge does not expose hidden testcase contents. Do not attempt to retrieve or infer them.

## Bootstrap

```bash
bash scripts/bootstrap_skills.sh
bash scripts/doctor.sh
```

The official skill currently provides `cannjudge_cli.py`, `generate_key.py`, and `encrypt_password.py`. Generate the key pair using the official script and keep the private key on the Ascend server.

## Credential setup

Preferred non-interactive setup uses a ciphertext file outside the repository:

```bash
export CANNJUDGE_EMAIL='you@example.com'
export CANNJUDGE_PRIVATE_KEY="$HOME/.config/xingchenbei/cannjudge/private.pem"
export CANNJUDGE_CIPHERTEXT_FILE="$HOME/.config/xingchenbei/cannjudge/password.rsa"
chmod 600 "$CANNJUDGE_PRIVATE_KEY"
```

`CANNJUDGE_CIPHERTEXT` is also supported for short-lived shell environments. The adapter never prints its value. When run directly in an interactive terminal, missing email/ciphertext values can be entered at prompts.

## MhcExpand: immediate use

The tracked task config is `config/tasks/mhc_expand.env` and contains problem 301's verified internal id plus the dual-track gate commands.

First check wiring without submitting:

```bash
python3 tools/cannjudge_eval.py doctor --task mhc_expand
```

Direct platform submission (explicit external side effect):

```bash
python3 tools/cannjudge_eval.py submit --task mhc_expand --yes-submit
```

Recommended gated submission:

```bash
python3 tools/agent_loop.py platform \
  --task mhc_expand \
  --submit \
  --name correctness-baseline
```

The gated path executes `guard -> platform 910B build -> local A3 validation -> CANNJudge submit/poll`. To intentionally reuse already-proven local gates, explicit skip flags exist, for example:

```bash
python3 tools/agent_loop.py platform \
  --task mhc_expand \
  --submit \
  --skip-build \
  --skip-validate \
  --name platform-only-check
```

Use skip flags only when you already have fresh evidence for those gates.

## Query and rank

```bash
python3 tools/cannjudge_eval.py query \
  --task mhc_expand \
  --submission-id '<submission-id>'

python3 tools/cannjudge_eval.py rank --task mhc_expand
```

## Evidence and decisions

A platform run records Git metadata, source SHA-256 values, submission ID, returned status/result, and score when available. Credentials are never included. An `Accepted` result with a numeric score is compared against `best-platform.json`; local benchmark scores use `best-local.json` and never overwrite platform bests.

Do not equate local A3 correctness with platform 910B correctness. For MhcExpand, the intended evidence labels remain `PLATFORM 910B BUILD PASS` and `LOCAL A3 CORRECTNESS PASS` until CANNJudge returns a real platform result.
