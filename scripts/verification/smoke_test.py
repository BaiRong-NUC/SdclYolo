from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import _bootstrap
from sdcl.cli import main
import sys

if __name__ == "__main__":
    raise SystemExit(main(["smoke", *sys.argv[1:]]))

