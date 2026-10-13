"""One frozen-feature classification screen; default recipe cannot be overridden."""

from pathlib import Path
import os
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import argparse
import json
from sdcl.training.class_margin_probe import prepare, train
from sdcl.postprocess.class_margin_probe import CACHE, RUN


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='0')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--cache',default=CACHE)
    parser.add_argument('--output',default=RUN)
    parser.add_argument('--smoke-limit',type=int,help='Images per fit/monitor group; smoke only')
    args=parser.parse_args()
    result=prepare(args.cache,args.device,args.smoke_limit)
    if not args.prepare_only:
        result=train(args.cache,args.output,args.device,args.resume)
    print(json.dumps(result,indent=2))
