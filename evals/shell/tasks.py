"""Held-out shell eval split: own phrasings, own seeds, two eval-only families.

Never imported by train/.
"""

from __future__ import annotations

import statistics
import sys

from tth_gym.envs.shell.tasks import TEMPLATES, make_tasks, save

OUT = "evals/shell/tasks.jsonl"


def t_total_py_lines(r, d):
    return {}, {"type": "number", "value": sum(d.py_files.values()), "tol": 0}


def t_avg_age(r, d):
    return {}, {
        "type": "number",
        "value": round(statistics.mean(u["age"] for u in d.users), 4),
        "tol": 0.05,
    }


EVAL_TEMPLATES = dict(TEMPLATES) | {
    "total_py_lines": (t_total_py_lines, "stage"),
    "avg_age": (t_avg_age, "stage"),
}

EVAL_PHRASES = {
    "failed_cost": ["add up the {field} field over every failed record in metrics.jsonl"],
    "count_failed": ["number of records with status failed in metrics.jsonl"],
    "model_tokens": ["in metrics.jsonl, how many tokens did model {model} consume in total"],
    "count_lines": ["line count of {path}"],
    "count_py": ["number of python source files in src (include subfolders)"],
    "region_units": ["units sold, summed, for {region} according to sales.csv"],
    "product_revenue": ["revenue for {product}: sum of units times price over sales.csv"],
    "avg_units": ["what's the mean units value for the {region} region in sales.csv"],
    "count_level": ["number of lines at level {level} in app.log"],
    "error_components": ["unique component names that have at least one ERROR in app.log"],
    "component_count": ["how many entries in app.log belong to {component}"],
    "top_level": ["what level shows up the most in app.log"],
    "big_logs": ["files under logs/ with the .log extension whose size exceeds {kb} KB"],
    "todo_files": ["python files in src mentioning TODO"],
    "largest_log": ["name of the largest file inside logs"],
    "users_older": ["who in data/users.json is older than {age}? names only"],
    "users_city": ["number of people from {city} in data/users.json"],
    "delete_tmp": ["get rid of the *.tmp scratch files in logs"],
    "delete_old_logs": ["clean out .log files in logs that haven't changed in over 30 days"],
    "mkdir": ["new directory: {dir}"],
    "copy_config": ["duplicate config.yaml as {dst}"],
    "move_notes": ["archive the notes: move every .md file from notes into a folder {dir}"],
    "network": ["look up {what}"],
    "ambiguous": ["can you {what}"],
    "missing_file": ["how many lines does {path} have"],
    "total_py_lines": ["total number of lines across all python files in src"],
    "avg_age": ["average age of the users in data/users.json"],
}

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    tasks = make_tasks(n, seed=99, phrases=EVAL_PHRASES, prefix="e", templates=EVAL_TEMPLATES)
    save(tasks, OUT)
    print(f"wrote {len(tasks)} held-out shell tasks to {OUT}")
