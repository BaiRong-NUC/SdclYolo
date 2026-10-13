from pathlib import Path
import os
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import argparse
import json
from sdcl.evaluation.class_margin_probe import evaluate
from sdcl.postprocess.class_margin_probe import RUN


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Fixed epoch10 visual classification probe')
    parser.add_argument('--checkpoint',default=f'{RUN}/last.pt')
    parser.add_argument('--output')
    parser.add_argument('--device',default='0')
    parser.add_argument('--octave',default='output/tools/octave/octave-11.3.0-w64/mingw64/bin/octave-cli.exe')
    parser.add_argument('--limit',type=int,help='Smoke only; never an investment decision')
    print(json.dumps(evaluate(**vars(parser.parse_args())),indent=2))
