#!/usr/bin/env python3
import difflib

def diff_ir(raw_text: str, opt_text: str, context=3):
    raw_lines = raw_text.splitlines()
    opt_lines = opt_text.splitlines()
    diff = difflib.unified_diff(
        raw_lines, opt_lines,
        fromfile="Raw Relay IR", tofile="Optimized Relay IR",
        lineterm="", n=context
    )
    return "\n".join(diff)
