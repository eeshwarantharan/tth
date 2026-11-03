# Security policy

tth is an early research preview. It has [known safety bypasses](docs/analysis/2026-11-audit.md#4-safety-defects) that we're fixing, so always read a staged command before running it.

## What to report privately

Use [GitHub private vulnerability reporting](https://github.com/eeshwarantharan/tth/security/advisories/new) for:

- Escaping the sandbox: writes, network access or secret reads from a probe
- tth executing a staged command
- Secrets reaching an expert payload, or an escalation without consent

Please include a minimal reproduction, your OS and your tth commit.

## What can be a normal issue or PR

A command that gets the wrong safety tier is welcome as a public PR while the project is in preview. Add it to `tests/safety/cases/`.

## Scope

The sandbox limits what probes can do during verification. It doesn't protect a command you choose to run yourself, which is why the tier badge matters and why you should always read the command first.
