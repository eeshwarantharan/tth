"""Shell widgets. They put a command on the prompt, and never run it."""

ZSH = r"""# tth shell integration (zsh). Type a request, press Ctrl-G: the command lands on your prompt.
_tth_widget() {
  local req="$BUFFER" out
  [[ -z "$req" ]] && return
  out=$(mktemp -t tth.XXXXXX) || return
  zle -I
  command tth run --out "$out" -- "$req" </dev/tty >/dev/tty 2>&1
  if [[ -s "$out" ]]; then BUFFER="$(<"$out")"; CURSOR=${#BUFFER}; fi
  rm -f "$out"
  zle reset-prompt
}
zle -N _tth_widget
bindkey '^G' _tth_widget
# Or: t <request>  -> the command appears on your next prompt (print -z), never executed.
t() {
  local out c; out=$(mktemp -t tth.XXXXXX) || return
  command tth run --out "$out" -- "$*"
  c="$(<"$out")"; rm -f "$out"
  [[ -n "$c" ]] && print -z -- "$c"
}
"""

BASH = r"""# tth shell integration (bash). Type a request, press Ctrl-G: the line is replaced by the command.
_tth_readline() {
  local req="$READLINE_LINE" out
  [[ -z "$req" ]] && return
  out=$(mktemp -t tth.XXXXXX) || return
  command tth run --out "$out" -- "$req" </dev/tty >/dev/tty 2>&1
  if [[ -s "$out" ]]; then READLINE_LINE="$(<"$out")"; READLINE_POINT=${#READLINE_LINE}; fi
  rm -f "$out"
}
bind -x '"\C-g": _tth_readline'
"""
