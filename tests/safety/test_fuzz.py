"""Adversarial composition: hide dangerous payloads inside every shell construct.

Deterministic, so a failure always reproduces. A payload in any position must never be classified READ_ONLY.
"""

from __future__ import annotations

import itertools

import pytest

from tth.safety.classifier import Tier, classify
from tth.safety.rewriter import rewrite

PAYLOADS = [
    "rm -rf build",
    "rm x",
    "unlink x",
    "sudo ls",
    "dd if=/dev/zero of=x",
    "curl -s http://x",
    "touch pwned",
    "python3 -c 'print(1)'",
    "sh -c ls",
    "mkdir pwned",
    "cp a b",
    "mv a b",
    "git reset --hard",
    "find . -delete",
    "sed -i s/a/b/ f",
]

CONTEXTS = [
    "{p}",
    "({p})",
    "{{ {p}; }}",
    "! {p}",
    "{p} &",
    "{p} | cat",
    "cat f | {p}",
    "true && {p}",
    "false || {p}",
    "ls; {p}",
    "$({p})",
    "`{p}`",
    "echo $({p})",
    "echo `{p}`",
    'echo "$({p})"',
    'echo "a$({p})b"',
    "echo ${{x:-$({p})}}",
    "echo ${{x:=$({p})}}",
    'echo "${{x:-$({p})}}"',
    "echo $(( $({p}) ))",
    "echo $(echo $({p}))",
    "cat <({p})",
    "X=$({p}) ls",
    'X="$({p})" ls',
    "cat <<EOF\n$({p})\nEOF",
    "cat <<EOF\n${{x:-$({p})}}\nEOF",
    "cat <<EOF | {p}\nhi\nEOF",
    "cat <<EOF && {p}\nhi\nEOF",
    "cat <<EOF; {p}\nhi\nEOF",
    'cat <<< "$({p})"',
    "cat < <({p})",
    "if {p}; then ls; fi",
    "if true; then {p}; fi",
    "for i in a; do {p}; done",
    "for i in $({p}); do ls; done",
    "while {p}; do ls; done",
    "case x in x) {p};; esac",
    "f() {{ {p}; }}; f",
    "ls $({p})",
    "find . $({p})",
    "cat $({p}) f",
    "echo x > $({p})",
    "ls | tee $({p})",
    "[[ -n $({p}) ]]",
    "time {p}",
    "env {p}",
    "command {p}",
    "nohup {p}",
    "nice {p}",
    "timeout 5 {p}",
]

NAME_OBFUSCATION = [
    "r''m x",
    'r""m x',
    "\\rm x",
    '"rm" x',
    "'rm' x",
    "$'rm' x",
    "$'r\\x6d' x",
    "/bin/rm x",
    "/usr/bin/../bin/rm x",
    "/bin/./rm x",
    "$(echo rm) x",
    "`echo rm` x",
    "$CMD x",
    "${CMD} x",
    "r${x}m x",
]


@pytest.mark.parametrize(("payload", "context"), list(itertools.product(PAYLOADS, CONTEXTS)))
def test_payload_is_never_read_only(payload: str, context: str) -> None:
    command = context.format(p=payload)
    c = classify(command)
    assert c.tier != Tier.READ_ONLY, f"read-only: {command!r}"


@pytest.mark.parametrize(("payload", "context"), list(itertools.product(PAYLOADS, CONTEXTS)))
def test_destructive_payload_is_never_staged_raw(payload: str, context: str) -> None:
    if not payload.startswith(("rm ", "unlink ", "git reset", "find . -delete", "dd ")):
        return
    command = context.format(p=payload)
    assert rewrite(command).staged != command, f"staged raw: {command!r}"


@pytest.mark.parametrize("command", NAME_OBFUSCATION)
def test_obfuscated_names_are_never_read_only(command: str) -> None:
    assert classify(command).tier != Tier.READ_ONLY, command
    assert rewrite(command).staged != command or classify(command).tier >= Tier.NEEDS_REVIEW, command
