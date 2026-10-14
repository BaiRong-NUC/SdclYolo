"""Archive public primary-source PDFs cited by the research notes, without overwrites."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import urllib.request

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "output/literature"
CATALOG = ROOT / "docs/literature/reference_catalog_20261014.json"
URL_PATTERN = r"https?://[^\s`<>\"\\)]+"
EXTRA_ARXIV = (
    "1704.04503", "1705.02950", "1711.11575", "1904.03629", "2202.06934",
    "2505.18446", "2409.14679", "2211.11426", "2305.15354", "2108.07755",
)
EXTRA_PDF = (
    "https://openaccess.thecvf.com/content/CVPR2022/papers/"
    "Liu_Contextual_Debiasing_for_Visual_Recognition_With_Causal_Mechanisms_CVPR_2022_paper.pdf",
    "https://openaccess.thecvf.com/content_CVPR_2020/papers/"
    "Singh_Dont_Judge_an_Object_by_Its_Context_Learning_to_Overcome_CVPR_2020_paper.pdf",
)
EXISTING = {
    "arxiv-1807.11590": "output/verification/ranking_nearest_prior_20261013/iou_net.pdf",
    "cvf-Liu_RankDetNet_Delving_Into_Ranking_Constraints_for_Object_Detection_CVPR_2021_paper":
        "output/verification/ranking_nearest_prior_20261013/rankdetnet.pdf",
    "cvf-Tan_Learning_to_Rank_Proposals_for_Object_Detection_ICCV_2019_paper":
        "output/verification/ranking_nearest_prior_20261013/rank_nms.pdf",
}


def normalize(url):
    if "arxiv.org/" in url:
        match = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", url)
        if match:
            identifier = match[1]
            return "arxiv-" + identifier, "https://arxiv.org/pdf/" + identifier
    if "thecvf.com/" in url:
        pdf = url.replace("/html/", "/papers/").replace(".html", ".pdf")
        return "cvf-" + Path(pdf).stem, pdf
    if "proceedings.iclr.cc/" in url:
        pdf = url.replace("/hash/", "/file/").replace("-Abstract-Conference.html", "-Paper-Conference.pdf")
        return "iclr-" + Path(pdf).stem, pdf
    if "doi.org/" in url:
        return "doi-" + url.split("doi.org/")[-1].replace("/", "_"), None
    return None, None


def catalog():
    entries = {}
    for path in (ROOT / "docs").rglob("*"):
        if path.suffix not in (".md", ".json") or "literature" in path.parts:
            continue
        text = path.read_text(encoding="utf-8-sig")
        for url in re.findall(URL_PATTERN, text):
            key, pdf = normalize(url)
            if key is None:
                continue
            item = entries.setdefault(key, {"id": key, "pdf_url": pdf, "sources": [], "cited_by": []})
            for field, value in (("sources", url), ("cited_by", path.relative_to(ROOT).as_posix())):
                if value not in item[field]:
                    item[field].append(value)
    for identifier in EXTRA_ARXIV:
        key, url = normalize("https://arxiv.org/abs/" + identifier)
        entries.setdefault(key, {"id": key, "pdf_url": url,
                                "sources": ["https://arxiv.org/abs/" + identifier],
                                "cited_by": ["additional prior-art screen"]})
    for url in EXTRA_PDF:
        key, pdf = normalize(url)
        entries.setdefault(key, {"id": key, "pdf_url": pdf, "sources": [url],
                                "cited_by": ["context intervention prior-art screen"]})
    entries.setdefault("doi-10.1145_3447548.3467121", {
        "id": "doi-10.1145_3447548.3467121",
        "pdf_url": "https://dl.acm.org/doi/pdf/10.1145/3447548.3467121",
        "sources": ["https://doi.org/10.1145/3447548.3467121"],
        "cited_by": ["docs/960_class_conflict_prior_audit_20261014.md"],
        "verified_title": "Robust Object Detection Fusion Against Deception",
        "verified_authors": ["Ka-Ho Chow", "Ling Liu"], "verified_year": 2021,
        "metadata_source": "Paper Search Crossref search, exact DOI/title match, 2026-10-14",
    })
    entries.setdefault("doi-10.1049_cit2.70083", {
        "id": "doi-10.1049_cit2.70083",
        "pdf_url": "https://ietresearch.onlinelibrary.wiley.com/doi/pdfdirect/10.1049/cit2.70083",
        "sources": ["https://doi.org/10.1049/cit2.70083",
                    "https://github.com/cszzshi/DALA"],
        "cited_by": ["docs/960\u8986\u76d6\u4fdd\u62a4\u6807\u7b7e\u5206\u914d\u4e00\u6b21\u7b5b\u9009.md"],
        "verified_title": "Dynamic Adaptive Label Assignment for Tiny Object Detection in Remote Sensing Images",
        "metadata_source": "Author repository citation, current archive",
    })
    return {"date": "2026-10-14", "scope": "Cited primary papers plus current-context nearest work; "
            "downloading does not mean full method review or originality confirmation.",
            "papers": sorted(entries.values(), key=lambda item: item["id"])}


def inspect_pdf(path):
    if path.read_bytes()[:5] != b"%PDF-":
        raise ValueError("Response is not a PDF.")
    doc = PdfReader(path)
    if doc.is_encrypted or not len(doc.pages):
        raise ValueError("Unreadable or encrypted PDF.")
    text = "\n".join(f"\n--- PAGE {i + 1} ---\n" + (page.extract_text() or "")
                     for i, page in enumerate(doc.pages))
    surrogates = sum(0xD800 <= ord(c) <= 0xDFFF for c in text)
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    metadata = doc.metadata or {}
    return {"pages": len(doc.pages), "metadata_title": metadata.get("/Title", ""),
            "metadata_author": metadata.get("/Author", ""),
            "unmapped_surrogate_glyphs_replaced_in_text_only": surrogates}, text


def obtain(item):
    identifier = item["id"]
    target = DESTINATION / "papers" / (identifier + ".pdf")
    result = {"id": identifier, "source_url": item["pdf_url"],
              "checked_at_utc": datetime.now(timezone.utc).isoformat()}
    try:
        route = "existing_public_pdf"
        if not target.exists():
            if identifier in EXISTING:
                shutil.copy2(ROOT / EXISTING[identifier], target)
                route = "verified_historical_archive_copy"
            elif identifier.startswith("arxiv-") and (
                    DESTINATION / "papers" / (identifier[6:] + ".pdf")).exists():
                shutil.copy2(DESTINATION / "papers" / (identifier[6:] + ".pdf"), target)
                route = "Paper Search MCP download"
            elif item["pdf_url"] is None:
                return {**result, "status": "needs_open_access_location",
                        "reason": "DOI identified; no verified public PDF URL yet."}
            else:
                request = urllib.request.Request(item["pdf_url"], headers={"User-Agent": "ResearchArchive/1.0"})
                with urllib.request.urlopen(request, timeout=25) as response:
                    data = response.read(50 * 1024 * 1024 + 1)
                    final_url = response.url
                if len(data) > 50 * 1024 * 1024 or not data.startswith(b"%PDF-"):
                    raise ValueError("Not a PDF or larger than the 50 MiB archive limit.")
                partial = target.with_suffix(".partial")
                partial.write_bytes(data)
                inspect_pdf(partial)
                partial.replace(target)
                route = "public_primary_pdf_http"
                result["resolved_url"] = final_url
        details, text = inspect_pdf(target)
        text_path = DESTINATION / "text" / (identifier + ".txt")
        if not text_path.exists() or text_path.stat().st_size == 0:
            text_path.write_text(text, encoding="utf-8")
        return {**result, "status": "downloaded_and_readable", "route": route,
                "path": str(target), "bytes": target.stat().st_size,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), **details,
                "text_path": str(text_path),
                "text_sha256": hashlib.sha256(text_path.read_bytes()).hexdigest(),
                "review_status": "text extracted, method review not inferred"}
    except Exception as error:
        return {**result, "status": "download_unavailable", "error": str(error)}


def run(workers=2):
    DESTINATION.mkdir(parents=True, exist_ok=True)
    (DESTINATION / "papers").mkdir(exist_ok=True)
    (DESTINATION / "text").mkdir(exist_ok=True)
    CATALOG.parent.mkdir(parents=True, exist_ok=True)
    value = catalog()
    CATALOG.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    manifest_path = DESTINATION / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"papers": {}}
    manifest["catalog_sha256"] = hashlib.sha256(CATALOG.read_bytes()).hexdigest()
    manifest.setdefault("zotero", {"status": "local_api_unavailable_at_last_check",
                                  "no_library_writes_completed": True})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(obtain, item) for item in value["papers"]]
        for future in as_completed(futures):
            item = future.result()
            manifest["papers"][item["id"]] = item
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({k: item[k] for k in ("id", "status")}), flush=True)
    counts = {}
    for item in manifest["papers"].values():
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    snapshot = {"date": "2026-10-14", "counts": counts,
                "catalog_sha256": manifest["catalog_sha256"], "papers": manifest["papers"],
                "zotero": manifest["zotero"]}
    (CATALOG.parent / "download_status_20261014.json").write_text(
        json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": counts, "manifest": str(manifest_path)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    run(**vars(parser.parse_args()))
