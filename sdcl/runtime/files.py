"""Atomic experiment writes with bounded retries for Windows sharing locks."""

import os
from time import sleep


def atomic_replace(source, target):
    for attempt in range(6):
        try:
            source.replace(target)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 5:
                raise
            # Antivirus/indexing may briefly hold a newly written manifest.
            sleep(0.1 * (attempt + 1))
