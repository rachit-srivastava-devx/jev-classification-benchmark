"""Loading the candidate corpus, whichever way it is stored on disk.

The corpus is 34.8 MB of raw JSON and it is committed, because a benchmark whose
inputs are not in the repository cannot be re-run by a reader. Gzipped it is
9.8 MB, which is the difference between a repository that clones quickly and one
that does not, so the committed artifact is `tasks.json.gz`.

Both forms stay readable here rather than at the call sites: the runner and the
report builder each load the corpus, and a second copy of this three-line
decision is how the two would eventually disagree about which file is canonical.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path


def load_tasks(path: str | Path) -> dict:
    """Read the corpus from `path`, transparently un-gzipping if needed.

    `path` may name either form. If it names the plain `.json` and only the
    `.json.gz` exists, the gzipped one is used: that way a command line written
    before the file was compressed keeps working instead of failing with a
    confusing "no such file".

    Raises FileNotFoundError naming both candidates if neither exists, rather
    than the bare one-path message, which sends the reader looking for the
    wrong file.
    """
    path = Path(path)
    if path.suffix == ".gz":
        return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    if path.exists():
        return json.loads(path.read_text())

    packed = path.with_suffix(path.suffix + ".gz")
    if packed.exists():
        return json.loads(gzip.decompress(packed.read_bytes()).decode("utf-8"))

    raise FileNotFoundError(f"found neither {path} nor {packed}")
