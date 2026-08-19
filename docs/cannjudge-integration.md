# CANNJudge integration

The harness bootstraps the official CANN `cann-learning-hub` repository and exposes:

- `cannjudge-submit`
- `ascendc-ops-project` (supporting dependency/reference)

under `.agents/skills/`.

## Responsibility

Use `cannjudge-submit` for CANNJudge platform interaction:

- obtain current problem metadata;
- download an official problem package for comparison;
- submit a candidate only when explicitly authorized;
- query submission status/results;
- inspect rankings when useful.

Use the dedicated `Ascend/agent-skills` skills for core Host/Tiling/Kernel design, code generation, debugging, precision work, and performance optimization.

## Credential safety

The official CANNJudge skill supports an RSA-encrypted login flow. Follow these rules:

1. Never paste a plaintext CANNJudge password into Codex/chat.
2. Generate the RSA key pair locally on the Ascend/server environment as instructed by the official skill.
3. Keep `private.pem` local and never print or commit it.
4. Do not commit `public.pem`, credential ciphertext, tokens, cookies, or session data either.
5. Do not echo decrypted credentials into logs.
6. CANNJudge submission is an external side effect; require explicit user authorization before submitting.

The repository `.gitignore` excludes generated dependency checkouts, skill symlinks, and common key/credential filenames.

## Platform evidence

A current CANNJudge response and freshly downloaded official package for a verified matching problem ID are strong platform evidence. Compare them with the checked-in task statement/template; do not silently overwrite local artifacts when they differ.

CANNJudge intentionally does not expose hidden testcase contents. Do not attempt to retrieve them or optimize by hidden-case guessing; preserve shape/dtype/attribute/alignment/boundary generality.

## Bootstrap

```bash
bash scripts/bootstrap_skills.sh
bash scripts/doctor.sh
```

After bootstrap, Codex should discover `$cannjudge-submit` and `$ascendc-ops-project` alongside the Ascend skills.
