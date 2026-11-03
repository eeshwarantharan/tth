# Security policy

tth is an early research preview and hasn't been independently reviewed, so always read a staged command before running it. The [audit](docs/analysis/2026-11-audit.md#4-safety-defects) records what has been found and fixed so far.

## Known limits

- **Allowlist depth.** Rules cover the flags we know about. A new or unusual flag on an allowed tool could still write or execute.
- **`tar` extraction** can overwrite existing files, and is only badged as changing files.
- **Repository config.** Running git in a repository you don't trust can execute programs through its config, even for read-only subcommands.
- **macOS sandbox.** The `sandbox-exec` profile starts from allow-by-default, then denies writes, network and secret reads. It isn't a default-deny sandbox.
- **Redaction** is pattern-based. Review the payload shown at the consent prompt.

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
