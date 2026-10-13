"""Compile and verify native helpers before running original VisDrone aggregation."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from sdcl.config import project_path
from sdcl.evaluation.official_visdrone import (
    ARMS, collect_results, matlab_string, save_json, verify_prepared,
)
from sdcl.evaluation.size_analysis import file_hash


HELPERS = ("createIntImg", "compOas", "evalRes", "VOCap")
REFERENCES = (*HELPERS, "dropObjectsInIgr", "saveAnnoRes", "calcAccuracy")


def octave_env(octave):
    env = os.environ.copy()
    env["PATH"] = str(Path(octave).resolve().parent) + os.pathsep + env.get("PATH", "")
    return env


def build_runtime(output, octave):
    output = project_path(output)
    verify_prepared(output)
    runtime = output / "runtime"
    if runtime.exists():
        raise FileExistsError("Refusing to overwrite runtime verification.")
    runtime.mkdir()
    source = project_path("sdcl/evaluation/native/visdrone_helpers.cc")
    shutil.copyfile(source, runtime / source.name)
    shutil.copyfile(Path(__file__), runtime / "runtime_source.py")
    references = runtime / "references"
    references.mkdir()
    for name in REFERENCES:
        text = (output / "toolkit/utils" / f"{name}.m").read_text()
        for old in REFERENCES:
            text = re.sub(r"\b" + old + r"\b", "ref_" + old, text)
        (references / f"ref_{name}.m").write_text(text, encoding="ascii")
    library = runtime / "visdrone_helpers.oct"
    command = [
        str(Path(octave).resolve().parent / "mkoctfile.exe"),
        "-O2", "-ffp-contract=off", "-o", str(library), str(runtime / source.name),
    ]
    with (runtime / "compile.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, env=octave_env(octave), stdout=log,
                                   stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError("Native helper build failed; inspect runtime/compile.log.")
    root, lib = matlab_string(output), matlab_string(library)
    setup = "\n".join(f"autoload('{name}', '{lib}');" for name in HELPERS)
    (runtime / "setup_helpers.m").write_text(setup + "\n", encoding="ascii")
    (runtime / "validate.m").write_text(validation_code(root), encoding="ascii")
    (runtime / "run_fast.m").write_text(
        f"root = '{root}';\nrun(fullfile(root,'runtime','setup_helpers.m'));\n"
        "run(fullfile(root,'run_official.m'));\n", encoding="ascii"
    )
    save_json(runtime / "manifest.json", {
        "helpers": list(HELPERS),
        "scope": "Numeric arrays only. Original calcAccuracy/saveAnnoRes/dropObjectsInIgr remain unchanged.",
        "reference_namespacing": "Only function identifiers were prefixed; original reference logic is unchanged.",
        "octave_path": str(Path(octave).resolve()), "octave_executable_sha256": file_hash(octave),
        "files": {str(p.relative_to(output)).replace("\\", "/"): file_hash(p)
                  for p in sorted(runtime.rglob("*")) if p.is_file() and p.name != "manifest.json"},
        "runner_hashes": {str(p.relative_to(output)).replace("\\", "/"): file_hash(p)
                          for p in (output / "run_official.m", output / "compat/mean2.m")},
    })
    return runtime


def validate_runtime(output, octave):
    output = project_path(output)
    verify_runtime_files(output)
    runtime = output / "runtime"
    if (runtime / "validation.json").exists():
        raise FileExistsError("Refusing to overwrite validation.")
    with (runtime / "validate.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [str(Path(octave).resolve()), "--no-gui", "--quiet", "--no-history",
             str(runtime / "validate.m")],
            env=octave_env(octave), stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    if result.returncode or not (runtime / "validation.json").exists():
        raise RuntimeError("Native/helper equivalence failed; inspect runtime/validate.log.")
    verify_runtime_files(output)
    return json.loads((runtime / "validation.json").read_text())


def verify_runtime_files(output):
    output = project_path(output)
    verify_prepared(output)
    manifest = json.loads((output / "runtime/manifest.json").read_text())
    for name, digest in {**manifest["files"], **manifest["runner_hashes"]}.items():
        if file_hash(output / name) != digest:
            raise ValueError(f"Official runtime source/binary changed: {name}")
    if file_hash(manifest["octave_path"]) != manifest["octave_executable_sha256"]:
        raise ValueError("Octave executable changed.")
    return manifest


def run_accelerated(output, octave, arm):
    output = project_path(output)
    manifest = verify_runtime_files(output)
    if Path(octave).resolve() != Path(manifest["octave_path"]):
        raise ValueError("Use the exact validated Octave executable.")
    validation = json.loads((output / "runtime/validation.json").read_text())
    if not validation["passed"] or arm not in ARMS:
        raise ValueError("No validated runtime or unknown arm.")
    path = output / f"official_{arm}.log"
    if path.exists() or (output / f"official_{arm}.mat").exists():
        raise FileExistsError("Refusing to overwrite official result/log.")
    with path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [str(Path(octave).resolve()), "--no-gui", "--quiet", "--no-history",
             str(output / "runtime/run_fast.m"), arm],
            env=octave_env(octave), stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"Official evaluator failed; inspect {path}")
    verify_runtime_files(output)
    report = collect_results(output)
    report["execution"] = "Original official aggregation/preprocessing with cross-validated native hot-loop helpers."
    report["native_validation"] = validation
    save_json(output / "official_metrics.json", report)
    return report


def validation_code(root):
    return f"""root = '{root}';
addpath(fullfile(root,'toolkit','utils'));
addpath(fullfile(root,'compat'));
addpath(fullfile(root,'runtime','references'));
run(fullfile(root,'runtime','setup_helpers.m'));
rng(0);
integrals = 0; matches = 0; overlap_cases = 0;
for k=1:30
  image = double(rand(23,31)>.65);
  assert(isequal(createIntImg(image),ref_createIntImg(image)));
  image = randn(11,13);
  assert(isequal(createIntImg(image),ref_createIntImg(image)));
  integrals = integrals + 2;
end
for k=1:30
  ng = mod(k,13); nd = mod(3*k,37);
  g = [20*rand(ng,2), 1+20*rand(ng,2), double(rand(ng,1)>.7)];
  d = [20*rand(nd,2), 1+20*rand(nd,2), round(4*rand(nd,1))/4];
  if ng>0 && nd>0, d(1,1:4)=g(1,1:4); end
  if ng>1, g(2,1:4)=g(1,1:4); end
  assert(isequal(compOas(d(:,1:4),g(:,1:4),g(:,5)),ref_compOas(d(:,1:4),g(:,1:4),g(:,5))));
  overlap_cases=overlap_cases+1;
  for thr=.5:.05:.95
    for mul=[0,1]
      [g1,d1]=evalRes(g,d,thr,mul);
      [g2,d2]=ref_evalRes(g,d,thr,mul);
      assert(isequal(g1,g2) && isequal(d1,d2));
      matches=matches+1;
    end
  end
end
g = [0,0,10,10,1; 0,0,10,10,0; 0,0,10,10,0];
d = [0,0,10,10,.8; 0,0,10,10,.8; 0,0,10,10,.8; 0,0,10,10,.7];
for thr=[.5,.75,.95,1]
  [g1,d1]=evalRes(g,d,thr); [g2,d2]=ref_evalRes(g,d,thr);
  assert(isequal(g1,g2) && isequal(d1,d2)); matches=matches+1;
end
for fraction=[.5,.75,.95]
  g=[0,0,10,10,0]; d=[0,0,10*fraction,10,.6];
  [g1,d1]=evalRes(g,d,fraction); [g2,d2]=ref_evalRes(g,d,fraction);
  assert(isequal(g1,g2) && isequal(d1,d2)); matches=matches+1;
end
for k=0:40
  r=sort(rand(k,1)); p=rand(k,1);
  assert(abs(VOCap(r,p)-ref_VOCap(r,p)) <= 1e-12);
end
allgt=cell(1,3); alldet=cell(1,3);
for k=1:3
  categories=(1:10)';
  if k==2, categories=[1;4;4]; end
  if k==3, categories=[1;1;2;10]; end
  n=length(categories);
  allgt{{k}}=[10*rand(n,2),5+10*rand(n,2),zeros(n,1),categories,zeros(n,2)];
  alldet{{k}}=[allgt{{k}}(:,1:4), .3+.5*rand(n,1),categories, -ones(n,2)];
  alldet{{k}}=[alldet{{k}};alldet{{k}}];
  [~,ord]=sort(alldet{{k}}(:,5),'descend'); alldet{{k}}=alldet{{k}}(ord,:);
end
[a,b,c,d,e,f,g]=calcAccuracy(3,allgt,alldet); fast=[a,b,c,d,e,f,g];
[a,b,c,d,e,f,g]=ref_calcAccuracy(3,allgt,alldet); original=[a,b,c,d,e,f,g];
assert(max(abs(fast-original)) <= 1e-12);
gt=[0,0,20,20,0,0,0,0;15,0,20,20,0,0,0,0;5,5,15,15,1,1,0,0;35,25,4,5,1,2,0,0];
dt=[5.5,5.5,15.25,15.25,.8,1,-1,-1;35,25,4,5,.5,2,-1,-1];
[g1,d1]=dropObjectsInIgr(gt,dt,40,60);
[g2,d2]=ref_dropObjectsInIgr(gt,dt,40,60);
assert(isequal(g1,g2) && isequal(d1,d2));
names=findImageList(fullfile(root,'dataset','annotations'));
name=names{{1}};
gt=load(fullfile(root,'dataset','annotations',name));
dt=load(fullfile(root,'detections','calibration_no_gt',name));
img=imread(fullfile(root,'dataset','images',[name(1:end-4) '.jpg']));
[h,w,~]=size(img);
tic;
[g1,d1]=dropObjectsInIgr(gt,dt,h,w);
[g2,d2]=ref_dropObjectsInIgr(gt,dt,h,w);
assert(isequal(g1,g2) && isequal(d1,d2));
real_preprocessing_seconds=toc;
g1(g2(:,5)==0,5)=1; g1(g2(:,5)==1,5)=0;
for category=1:10
  g=g1(g1(:,6)==category,1:5); d=d1(d1(:,6)==category,1:5);
  for thr=[.5,.75,.95]
    [ga,da]=evalRes(g,d,thr); [gb,db]=ref_evalRes(g,d,thr);
    assert(isequal(ga,gb) && isequal(da,db)); matches=matches+1;
  end
end
verification=struct('passed',true,'octave_version',version,'integral_cases',integrals, ...
 'overlap_cases',overlap_cases,'matching_cases',matches,'precision_envelope_cases',41, ...
 'full_official_aggregation_fixture_max_error',max(abs(fast-original)), ...
 'union_ignore_fixture_exact',true,'real_image_preprocessing_exact',true, ...
 'real_image',name,'real_preprocessing_seconds',real_preprocessing_seconds);
fid=fopen(fullfile(root,'runtime','validation.json'),'w');
fprintf(fid,'%s\\n',jsonencode(verification)); fclose(fid);
disp(jsonencode(verification));
"""
