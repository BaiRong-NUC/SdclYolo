"""Expose class tables from the same official aggregation without changing it."""

import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.io import loadmat

from sdcl.config import project_path
from sdcl.evaluation.official_visdrone import ARMS, matlab_string, save_json
from sdcl.evaluation.official_runtime import octave_env, verify_runtime_files
from sdcl.evaluation.size_analysis import file_hash


NAMES = (
    "pedestrian", "people", "bicycle", "car", "van", "truck",
    "tricycle", "awning-tricycle", "bus", "motor",
)


def audit_details(output, octave):
    output = project_path(output)
    runtime = verify_runtime_files(output)
    if str(Path(octave).resolve()) != runtime["octave_path"]:
        raise ValueError("Use the validated runtime.")
    if not json.loads((output / "runtime/validation.json").read_text())["passed"]:
        raise ValueError("Native helper validation is required.")
    if any(not (output / f"official_{arm}.mat").exists() for arm in ARMS):
        raise ValueError("Run all four official evaluations first.")
    audit = output / "class_audit"
    if audit.exists():
        raise FileExistsError("Refusing to overwrite the class audit.")
    audit.mkdir()
    original = (output / "toolkit/utils/calcAccuracy.m").read_text()
    signature, body = original.split("\n", 1)
    expanded = signature.replace("AR_500]", "AR_500, AP, AR, evalClass]").replace(
        "calcAccuracy(", "inspectAccuracy("
    )
    if expanded == signature or "evalClass]" not in expanded:
        raise ValueError("Unknown upstream function signature.")
    (audit / "inspectAccuracy.m").write_text(expanded + "\n" + body, encoding="ascii")
    assert (audit / "inspectAccuracy.m").read_text().split("\n", 1)[1] == body
    root = matlab_string(output)
    arms = ",".join(f"'{arm}'" for arm in ARMS)
    script = f"""root='{root}';
addpath(fullfile(root,'toolkit','utils'));
addpath(fullfile(root,'compat'));
addpath(fullfile(root,'class_audit'));
run(fullfile(root,'runtime','setup_helpers.m'));
arms={{{arms}}};
for k=1:length(arms)
  arm=arms{{k}};
  data=load(fullfile(root,['preprocessed_' arm '.mat']));
  expected=load(fullfile(root,['official_' arm '.mat']));
  [a,b,c,d,e,f,g,AP,AR,evalClass]=inspectAccuracy(548,data.allgt,data.alldet);
  values=[a,b,c,d,e,f,g];
  assert(max(abs(values-expected.values)) <= 1e-12);
  class_images=zeros(10,1); class_gt=zeros(10,1); class_ignored_gt=zeros(10,1);
  for category=1:10
    class_images(category)=nnz(evalClass==category);
    for image=1:548
      gt=data.allgt{{image}};
      class_gt(category)=class_gt(category)+nnz(gt(:,6)==category);
      class_ignored_gt(category)=class_ignored_gt(category)+nnz(gt(:,6)==category & gt(:,5)==1);
    end
  end
  save('-mat7-binary',fullfile(root,'class_audit',[arm '.mat']), ...
       'AP','AR','evalClass','values','class_images','class_gt','class_ignored_gt');
  fprintf('CLASS_AUDIT_COMPLETED %s\\n',arm);
end
"""
    (audit / "audit.m").write_text(script, encoding="ascii")
    shutil_source = Path(__file__)
    (audit / "audit_source.py").write_bytes(shutil_source.read_bytes())
    with (audit / "audit.log").open("x", encoding="utf-8") as log:
        done = subprocess.run(
            [str(Path(octave).resolve()), "--no-gui", "--quiet", "--no-history",
             str(audit / "audit.m")],
            env=octave_env(octave), stdout=log, stderr=subprocess.STDOUT,
        )
    if done.returncode:
        raise RuntimeError("Official class audit failed; inspect class_audit/audit.log.")
    verify_runtime_files(output)
    report = {}
    for arm in ARMS:
        detail = loadmat(audit / f"{arm}.mat")
        expected = loadmat(output / f"official_{arm}.mat")["values"].ravel()
        error = float(np.max(np.abs(detail["values"].ravel() - expected)))
        if error > 1e-12:
            raise ValueError("Class audit differs from the original official call.")
        ap, ar = detail["AP"], detail["AR"]
        presence = detail["class_images"].ravel().astype(int)
        gt = detail["class_gt"].ravel().astype(int)
        ignored = detail["class_ignored_gt"].ravel().astype(int)
        report[arm] = {
            "headline_reproduction_max_error": error,
            "upstream_presence_weighted_AP_percent": float(expected[0]),
            "equal_class_macro_AP_percent_for_audit_only": float(ap.mean()),
            "classes": {
                name: {
                    "AP_percent": float(ap[i].mean()), "AP50_percent": float(ap[i, 0]),
                    "AP75_percent": float(ap[i, 5]), "AR500_percent": float(ar[i, :, 3].mean()),
                    "presence_images": int(presence[i]), "post_ignore_gt": int(gt[i]),
                    "ignore_flag_gt_in_upstream_denominator": int(ignored[i]),
                }
                for i, name in enumerate(NAMES)
            },
        }
    result = {
        "arms": report,
        "signature_only_extension": True,
        "official_function_body_unchanged": True,
        "aggregation_note": "The pinned upstream evalClass repeats classes by their image presence. Headline results preserve that weighting; the equal-class mean is a supplemental audit value.",
        "hashes": {str(p.relative_to(output)).replace("\\", "/"): file_hash(p)
                   for p in sorted(audit.iterdir()) if p.is_file()},
    }
    save_json(output / "official_class_audit.json", result)
    return result
