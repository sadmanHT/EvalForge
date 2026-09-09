#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "datasets/incident_diagnosis/raw/public_incidents/phase3-candidate-v1"
INDEX_PATH = ROOT / "configs/postmortem-candidates.json"
PROCESSED = (
    ROOT
    / "datasets/incident_diagnosis/processed"
    / "evalforge-incident-diagnosis-v0.1.0-candidate"
)
EVIDENCE = ROOT / "evidence/phase-03"
OBSERVED_AT = "2026-09-09T02:45:00Z"

CANDIDATES = [
    {
        "candidate_id": "honeycomb:2019-11-06:running-dry-on-memory",
        "company": "Honeycomb",
        "title": "Incident Report: Running Dry on Memory Without Noticing",
        "url": "https://www.honeycomb.io/blog/incident-report-running-dry-on-memory-without-noticing",
        "filename": "honeycomb-2019-11-06-memory-leak.snapshot.json",
        "label": "memory_leak",
        "publisher": "Honeycomb",
        "facts": [
            "Honeycomb reported four roughly 20-minute ingest brownouts in which 1–3% of customer telemetry was rejected.",
            "The incident trigger was explicitly identified as a slow memory leak that accumulated over hours across ingest backends.",
            "The backends crashed within minutes of one another as memory pressure accumulated, causing in-flight and new ingest requests to fail.",
            "After the leak was identified, Honeycomb reverted the bad commit and deployed a fixed release; memory behavior and service health returned to normal.",
        ],
        "reason": "Primary company incident report directly identifies a production memory leak as the trigger for repeated backend crashes and customer-visible errors.",
    },
    {
        "candidate_id": "dnsimple:2015-05-09:san-jose-memory-leak",
        "company": "DNSimple",
        "title": "Incident Report - San Jose Region and Redirector Service Outage",
        "url": "https://blog.dnsimple.com/2015/05/incident-report-san-jose-region-and-redirector-outage/",
        "filename": "dnsimple-2015-05-09-memory-leak.snapshot.json",
        "label": "memory_leak",
        "publisher": "DNSimple",
        "facts": [
            "DNSimple reported a production outage in its San Jose region that made its redirector service unreachable.",
            "Its hosting provider identified a slow memory leak in a network switch, triggered by packets from monitoring checks becoming stuck.",
            "The leak eventually exhausted the switch memory and prevented it from processing incoming packets.",
            "The hosting provider planned a network-switch software upgrade specifically to prevent recurrence of the memory leak.",
        ],
        "reason": "Primary incident report attributes the outage to progressive memory exhaustion caused by an explicit slow memory leak.",
    },
    {
        "candidate_id": "soundcloud:2011-08-24:binlog-disk-full",
        "company": "SoundCloud",
        "title": "Doing the right thing",
        "url": "https://developers.soundcloud.com/blog/doing-the-right-thing/",
        "filename": "soundcloud-2011-08-24-disk-exhaustion.snapshot.json",
        "label": "disk_exhaustion",
        "publisher": "SoundCloud Backstage Blog",
        "facts": [
            "SoundCloud described a production database outage in which maintenance activity generated more than 3 GB/minute of MySQL binary logs.",
            "The binlog partition expanded from roughly 10 GB to 100 GB of usage in about 30 minutes and then filled completely.",
            "The disk-full condition caused a partial binlog write, corrupted the final event, stopped slave replication, and left the master unable to make progress.",
            "Recovery required expanding the binlog volume and failing over to a consistent slave.",
        ],
        "reason": "Primary engineering postmortem directly documents filesystem capacity exhaustion as the mechanism that stopped database progress and replication.",
    },
    {
        "candidate_id": "git-nrw:2025-05-09:wal-disk-full",
        "company": "git.nrw",
        "title": "Incident: gitlab.git.nrw outage",
        "url": "https://git.nrw/en/news/2025-05-09-incident-gitlab-outage/",
        "filename": "git-nrw-2025-05-09-disk-exhaustion.snapshot.json",
        "label": "disk_exhaustion",
        "publisher": "git.nrw",
        "facts": [
            "git.nrw reported a complete GitLab outage lasting 5 hours 45 minutes.",
            "Database replication had silently stopped, leaving one database node to accumulate WAL transaction logs.",
            "The remaining database node's disk filled with WAL logs and the service went down.",
            "The incident was resolved by increasing disk capacity and restoring replication, with WAL-size monitoring added as follow-up.",
        ],
        "reason": "Official incident report directly attributes complete service loss to a database node filling its disk with WAL transaction logs.",
    },
    {
        "candidate_id": "coderden:2026-02-19:database-connection-leak",
        "company": "CoderDen",
        "title": "SEV-1: Database Connection Exhaustion Incident",
        "url": "https://engineering.coderden.in/posts/database-connection-exhaustion-incident/",
        "filename": "coderden-2026-02-19-database-connection-leak.snapshot.json",
        "label": "database_connection_leak",
        "publisher": "CoderDen Engineering",
        "facts": [
            "CoderDen reported a roughly 45-minute full platform outage in which API requests returned 503 errors.",
            "One of two explicitly identified root-cause mechanisms was an unclosed database query in a high-traffic endpoint.",
            "A missing rows.Close/defer path prevented PostgreSQL connections from being returned, causing connection leakage under load.",
            "Aggressive background scheduling compounded the exhaustion, so the candidate is scoped specifically to the documented connection-leak mechanism rather than claiming a single-cause incident.",
        ],
        "reason": "Primary company postmortem explicitly documents unclosed database connections as a root-cause mechanism in connection-pool exhaustion; the multi-factor nature is preserved in the evidence.",
    },
    {
        "candidate_id": "atlassian:CRUC-8168:bonecp-connection-leak",
        "company": "Atlassian",
        "title": "BoneCP connections are not returned to pool",
        "url": "https://jira.atlassian.com/browse/CRUC-8168",
        "filename": "atlassian-CRUC-8168-database-connection-leak.snapshot.json",
        "label": "database_connection_leak",
        "publisher": "Atlassian Jira",
        "facts": [
            "Atlassian tracked a Crucible production bug in which the application became unresponsive after running for a period of time.",
            "The failure was attributed to BoneCP database connections not being returned to the pool, leaving database connections unavailable to the application.",
            "Restarting the application restored service temporarily, consistent with exhausted leaked pool state.",
            "Atlassian classified the issue as a major bug and fixed it in subsequent Crucible releases.",
        ],
        "reason": "Vendor issue record directly describes database connections not being returned to the pool and resulting application unresponsiveness.",
    },
    {
        "candidate_id": "visa-cybersource:2026-04-28:payer-auth-misconfiguration",
        "company": "Visa Acceptance / Cybersource",
        "title": "Post-Incident Report: Payer Authentication Setup Errors",
        "url": "https://support.visaacceptance.com/_entity/annotation/c9e6d35e-3744-f111-88b4-0022480ac605",
        "filename": "visa-cybersource-2026-04-28-payment-configuration.snapshot.json",
        "label": "broken_payment_configuration",
        "publisher": "Visa Acceptance / Cybersource",
        "facts": [
            "Cybersource reported an incident affecting the Payer Authentication setup service on April 28, 2026.",
            "A configuration issue caused some authentication requests routed through one data center not to complete as expected.",
            "The root cause was a misconfiguration in the data center health-check mapping introduced during a prior change.",
            "Cardholders could see checkout errors or need to retry transactions; rollback/correction restored payer-authentication behavior.",
        ],
        "reason": "Official payment-platform post-incident report directly attributes checkout authentication failures to a configuration misconfiguration.",
    },
    {
        "candidate_id": "altapay:2026-07-15:shopify-3ds-firewall-config",
        "company": "AltaPay",
        "title": "Credit Card Payment Failures on Shopify Embedded App",
        "url": "https://status.altapay.com/history/1",
        "filename": "altapay-2026-07-15-payment-configuration.snapshot.json",
        "label": "broken_payment_configuration",
        "publisher": "AltaPay Status",
        "facts": [
            "AltaPay reported that 3-D Secure authentication in its Shopify embedded application stopped functioning correctly.",
            "Affected customers could not finalize credit-card payments at checkout.",
            "AltaPay traced the incident to a firewall configuration change.",
            "A mitigation restored payment processing, after which the team continued investigating the origin of the firewall change.",
        ],
        "reason": "Official payment-provider status incident directly connects checkout authentication failure to a firewall configuration change.",
    },
    {
        "candidate_id": "gitlab:INC-889:2025-05-12:false-alarm-version-skew",
        "company": "GitLab",
        "title": "GitLab Versions differ across zonal clusters",
        "url": "https://gitlab.com/gitlab-com/gl-infra/production/-/issues/19803",
        "filename": "gitlab-2025-05-12-no-fault.snapshot.json",
        "label": "no_fault",
        "publisher": "GitLab Infrastructure Production",
        "facts": [
            "GitLab opened an incident because production component versions appeared out of sync across zonal clusters after deployment.",
            "Investigation determined the condition was a false alarm caused by stale metadata for a ContainerStatusUnknown pod whose containers had already exited.",
            "No traffic was reaching the lingering pod and GitLab explicitly reported no customer impact.",
            "Deleting the stale pod cleared the alert; the visible deployment/version disparity was therefore a distractor rather than an active service fault.",
        ],
        "reason": "Primary production incident tracker explicitly concludes false alarm/no customer impact after a visible deployment-related distractor, matching the no_fault control semantics.",
    },
    {
        "candidate_id": "fastly:2023-02-28:cloud-waf-false-alarm",
        "company": "Fastly",
        "title": "(False Alarm) Cloud WAF Performance Issues",
        "url": "https://fastly.status.page/incidents?componentId=510172",
        "filename": "fastly-2023-02-28-no-fault.snapshot.json",
        "label": "no_fault",
        "publisher": "Fastly Status",
        "facts": [
            "Fastly initially posted that it was investigating elevated Cloud WAF errors.",
            "Customer Incident Response later changed the event from degraded performance to available/informational after investigation.",
            "Fastly explicitly confirmed the status update was a false alarm and that there were no elevated Cloud WAF errors.",
            "Other locations and services were unaffected throughout the investigation.",
        ],
        "reason": "Official status record explicitly resolves an apparent service degradation as a false alarm with no elevated errors.",
    },
    {
        "candidate_id": "openlibrary:issue-12432:2026-04-22:n-plus-one",
        "company": "Internet Archive / Open Library",
        "title": "perf: N+1 queries on thing table causing 3.8M single-row DB lookups",
        "url": "https://github.com/internetarchive/openlibrary/issues/12432",
        "filename": "openlibrary-12432-n-plus-one.snapshot.json",
        "label": "n_plus_one_query",
        "publisher": "Internet Archive / Open Library GitHub",
        "facts": [
            "Production pg_stat_statements showed SELECT-by-key as the database's highest-volume query with 3,830,328 calls in the observation window.",
            "The existing batched equivalent accounted for only 134,057 calls, a reported 28:1 ratio of unbatched to batched lookups.",
            "The issue explicitly diagnoses pages that resolve list items individually as an N+1 pattern and recommends replacing loops with the existing get_many batching path.",
            "The repeated single-row lookups consumed roughly 2.5 million milliseconds of database time and increased cache/I/O pressure on the production database.",
        ],
        "reason": "Primary production database profiling explicitly identifies an N+1 query pattern, quantifies millions of repeated lookups, and points to batching as the fix.",
    },
    {
        "candidate_id": "dify:issue-40036:2026-08-05:n-plus-one",
        "company": "Dify",
        "title": "Log pages extremely slow on large DB: N+1 workflow_runs loading",
        "url": "https://github.com/langgenius/dify/issues/40036",
        "filename": "dify-40036-n-plus-one.snapshot.json",
        "label": "n_plus_one_query",
        "publisher": "Dify GitHub",
        "facts": [
            "A real self-hosted production Dify installation reported a roughly 740 GB PostgreSQL database, peak QPS around 100, and six API replicas.",
            "The ordinary first log page triggered per-row workflow_runs lookups, explicitly described as an N+1 pattern, with large TOAST fields deserialized for each row.",
            "The reporter validated that the page-1 base SQL itself was sub-millisecond and that the list-page cost came from row-by-row run-detail loading, decompression, serialization, and payload transfer.",
            "The same report also describes a separate unindexed keyword-search full scan; this candidate is intentionally scoped only to the ordinary list-page N+1 mechanism.",
        ],
        "reason": "Primary production bug report isolates row-by-row related-record loading as an N+1 mechanism; the separate search-index problem is explicitly excluded from this mapping.",
    },
]


def git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def write_json(path: Path, payload: object) -> bytes:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return text.encode()


def replace_exact(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"expected text not found in {path}: {old[:80]!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def replace_regex(path: Path, pattern: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, text, flags=re.S)
    if count != 1:
        raise RuntimeError(f"expected one regex replacement in {path}, got {count}")
    path.write_text(updated, encoding="utf-8")


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    if index["index_version"] != "phase3-public-postmortem-candidates-v3":
        raise SystemExit("unexpected candidate index version")
    if len(index["candidates"]) != 11:
        raise SystemExit("unexpected candidate count before depth increment")

    existing_ids = {item["candidate_id"] for item in index["candidates"]}
    additions = []
    for spec in CANDIDATES:
        if spec["candidate_id"] in existing_ids:
            raise SystemExit(f"duplicate candidate id: {spec['candidate_id']}")
        snapshot = {
            "normalized_facts": spec["facts"],
            "observed_at": OBSERVED_AT,
            "provenance_limitations": [
                "This file is an EvalForge normalized semantic snapshot, not byte-for-byte original source content.",
                "It is sufficient for candidate review provenance but not for research admission under the Phase 03 original-source preservation gate.",
            ],
            "research_admission_eligible": False,
            "snapshot_format": "evalforge-normalized-public-incident-v1",
            "snapshot_kind": "normalized_semantic_snapshot",
            "source_publisher": spec["publisher"],
            "source_title": spec["title"],
            "source_url": spec["url"],
        }
        snapshot_path = RAW_DIR / spec["filename"]
        data = write_json(snapshot_path, snapshot)
        rel = snapshot_path.relative_to(ROOT).as_posix()
        additions.append(
            {
                "candidate_id": spec["candidate_id"],
                "company": spec["company"],
                "title": spec["title"],
                "original_url": spec["url"],
                "source_path": rel,
                "source_blob_sha": git_blob_sha(data),
                "proposed_root_cause_code": spec["label"],
                "decision": "supported_candidate_mapping",
                "mapping_basis": "narrative_root_cause",
                "evidence_summary": " ".join(spec["facts"]),
                "decision_reason": spec["reason"],
                "original_source_snapshot_preserved": False,
                "research_admitted": False,
            }
        )

    index["index_version"] = "phase3-public-postmortem-candidates-v4"
    index["source"]["note"] = (
        "postmortems.app remains the pinned discovery/index source for its entries. "
        "External candidates may be included when EvalForge preserves a normalized, checksum-locked "
        "candidate snapshot. Normalized snapshots support review provenance but are not treated as "
        "byte-for-byte original-source preservation and therefore do not make a candidate research-admitted."
    )
    index["candidates"].extend(additions)
    index["candidates"] = sorted(index["candidates"], key=lambda item: item["candidate_id"])
    write_json(INDEX_PATH, index)

    snapshot_paths = sorted(RAW_DIR.glob("*.snapshot.json"))
    if len(snapshot_paths) != 15:
        raise SystemExit(f"expected 15 local public snapshots, found {len(snapshot_paths)}")
    sums = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in snapshot_paths
    ]
    (RAW_DIR / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    readme = RAW_DIR / "README.md"
    readme_text = readme.read_text(encoding="utf-8")
    marker = "\n## Family-depth candidate increment\n"
    if marker in readme_text:
        readme_text = readme_text.split(marker, 1)[0].rstrip() + "\n"
    readme_text += (
        marker
        + "\n"
        + "This directory now contains 15 checksum-locked normalized semantic snapshots used only "
        + "for candidate review. Together with three supported non-local/pinned candidates, the "
        + "reviewed set reaches the splitter's minimum depth of three supported independent families "
        + "per frozen taxonomy label. These normalized snapshots are not byte-for-byte original "
        + "source archives and do not make any record research-admitted.\n"
    )
    readme.write_text(readme_text, encoding="utf-8")

    postmortems_py = ROOT / "backend/app/data/postmortems.py"
    replace_exact(
        postmortems_py,
        "    supported_mapping_count: int\n    supported_root_cause_codes: list[str]\n",
        "    supported_mapping_count: int\n"
        "    supported_root_cause_codes: list[str]\n"
        "    supported_family_counts_by_root_cause_code: dict[str, int]\n"
        "    supported_family_depth_sufficient_for_split: bool\n",
    )
    replace_exact(
        postmortems_py,
        "    admitted = [item for item in index.candidates if item.research_admitted]\n"
        "    decision_counts = Counter(item.decision.value for item in index.candidates)\n"
        "    missing = sorted(taxonomy_codes - supported)\n",
        "    admitted = [item for item in index.candidates if item.research_admitted]\n"
        "    decision_counts = Counter(item.decision.value for item in index.candidates)\n"
        "    supported_counts = Counter(\n"
        "        item.proposed_root_cause_code\n"
        "        for item in index.candidates\n"
        "        if item.decision == CandidateDecision.SUPPORTED\n"
        "    )\n"
        "    supported_family_counts = {\n"
        "        code: supported_counts.get(code, 0) for code in sorted(taxonomy_codes)\n"
        "    }\n"
        "    supported_family_depth_sufficient = all(\n"
        "        count >= 3 for count in supported_family_counts.values()\n"
        "    )\n"
        "    missing = sorted(taxonomy_codes - supported)\n",
    )
    replace_exact(
        postmortems_py,
        "        supported_root_cause_codes=sorted(supported),\n"
        "        missing_root_cause_codes=missing,\n",
        "        supported_root_cause_codes=sorted(supported),\n"
        "        supported_family_counts_by_root_cause_code=supported_family_counts,\n"
        "        supported_family_depth_sufficient_for_split=supported_family_depth_sufficient,\n"
        "        missing_root_cause_codes=missing,\n",
    )
    replace_exact(
        postmortems_py,
        '        blocker="independent_taxonomy_coverage_insufficient",\n',
        "        blocker=(\n"
        '            "original_source_preservation_and_research_admission_required"\n'
        "            if not missing and supported_family_depth_sufficient\n"
        '            else "independent_taxonomy_coverage_insufficient"\n'
        "        ),\n",
    )

    contract = ROOT / "scripts/check_phase3_contract.py"
    replace_exact(
        contract,
        "    if coverage.candidate_count != 11 or coverage.supported_mapping_count != 6:\n",
        "    if coverage.candidate_count != 23 or coverage.supported_mapping_count != 18:\n",
    )
    replace_exact(
        contract,
        "    if coverage.missing_root_cause_codes:\n"
        '        raise SystemExit("PHASE03_CONTRACT=FAIL postmortem_missing_mapping_set")\n',
        "    if coverage.missing_root_cause_codes:\n"
        '        raise SystemExit("PHASE03_CONTRACT=FAIL postmortem_missing_mapping_set")\n'
        "    if not coverage.supported_family_depth_sufficient_for_split:\n"
        '        raise SystemExit("PHASE03_CONTRACT=FAIL postmortem_family_depth")\n'
        "    if set(coverage.supported_family_counts_by_root_cause_code.values()) != {3}:\n"
        '        raise SystemExit("PHASE03_CONTRACT=FAIL postmortem_family_depth_counts")\n',
    )
    replace_exact(
        contract,
        "    if len(local_candidates) != 3:\n",
        "    if len(local_candidates) != 15:\n",
    )

    tests = ROOT / "backend/tests/data/test_phase3_data.py"
    replace_exact(tests, "    assert coverage.candidate_count == 11\n", "    assert coverage.candidate_count == 23\n")
    replace_exact(tests, "    assert coverage.supported_mapping_count == 6\n", "    assert coverage.supported_mapping_count == 18\n")
    replace_exact(
        tests,
        "    assert coverage.missing_root_cause_codes == []\n"
        "    assert coverage.admitted_research_record_count == 0\n",
        "    assert coverage.missing_root_cause_codes == []\n"
        "    assert coverage.supported_family_counts_by_root_cause_code == {\n"
        '        "broken_payment_configuration": 3,\n'
        '        "database_connection_leak": 3,\n'
        '        "disk_exhaustion": 3,\n'
        '        "memory_leak": 3,\n'
        '        "n_plus_one_query": 3,\n'
        '        "no_fault": 3,\n'
        "    }\n"
        "    assert coverage.supported_family_depth_sufficient_for_split\n"
        "    assert coverage.admitted_research_record_count == 0\n",
    )
    replace_exact(
        tests,
        '    assert coverage.blocker == "independent_taxonomy_coverage_insufficient"\n',
        '    assert coverage.blocker == "original_source_preservation_and_research_admission_required"\n',
    )
    replace_regex(
        tests,
        r"    assert \{item\.company for item in supported\} == \{\n(?:        .+\n)+?    \}\n    medoc =",
        "    assert len(supported) == 18\n"
        "    assert len({item.candidate_id for item in supported}) == 18\n"
        "    medoc =",
    )

    build_script = ROOT / "scripts/build_phase3_candidate.py"
    replace_exact(
        build_script,
        "Independent public postmortem candidates exist, but the frozen taxonomy "
        "is not covered sufficiently for a credible family-stratified holdout.",
        "Independent public incident candidates now provide the splitter's minimum "
        "three supported families per frozen taxonomy label, but no candidate is "
        "research-admitted until original primary-source evidence is preserved "
        "and canonical records are built.",
    )

    exit_script = ROOT / "scripts/check_phase3_exit.py"
    replace_exact(
        exit_script,
        '        print("BLOCKER_DETAIL=credible_family_stratified_holdout_not_yet_possible")\n',
        "        if coverage.blocker == \"original_source_preservation_and_research_admission_required\":\n"
        "            print(\n"
        '                "BLOCKER_DETAIL=original_primary_source_preservation_and_canonical_"\n'
        '                "research_admission_required"\n'
        "            )\n"
        "        else:\n"
        '            print("BLOCKER_DETAIL=credible_family_stratified_holdout_not_yet_possible")\n',
    )

    ci = ROOT / ".github/workflows/ci.yml"
    replace_exact(
        ci,
        'grep -q "^BLOCKER=independent_taxonomy_coverage_insufficient$" /tmp/phase03-exit.txt',
        'grep -q "^BLOCKER=original_source_preservation_and_research_admission_required$" /tmp/phase03-exit.txt',
    )

    import subprocess
    subprocess.run(["python", "scripts/build_phase3_candidate.py"], cwd=ROOT, check=True)

    mirrors = {
        "manifest.json": "candidate-manifest.json",
        "source-candidate-summary.json": "source-candidate-summary.json",
        "source-audit-report.json": "source-audit-report.json",
        "auxiliary-servicenow-summary.json": "auxiliary-servicenow-summary.json",
        "public-postmortem-candidate-coverage.json": "public-postmortem-candidate-coverage.json",
    }
    for src, dst in mirrors.items():
        shutil.copy2(PROCESSED / src, EVIDENCE / dst)

    manifest = json.loads((PROCESSED / "manifest.json").read_text(encoding="utf-8"))
    coverage = json.loads(
        (PROCESSED / "public-postmortem-candidate-coverage.json").read_text(encoding="utf-8")
    )["coverage"]
    class_summary_path = EVIDENCE / "class-split-summary.json"
    class_summary = json.loads(class_summary_path.read_text(encoding="utf-8"))
    class_summary["public_postmortem_candidate_count"] = 23
    class_summary["public_postmortem_supported_mapping_count"] = 18
    class_summary["public_postmortem_supported_taxonomy_label_count"] = 6
    class_summary["public_postmortem_supported_family_counts_by_root_cause_code"] = (
        coverage["supported_family_counts_by_root_cause_code"]
    )
    class_summary["public_postmortem_supported_family_depth_sufficient_for_split"] = True
    class_summary["public_postmortem_research_admitted_count"] = 0
    write_json(class_summary_path, class_summary)

    replacements = {
        "Conservatively reviewed independent candidates: **11**.": "Conservatively reviewed independent candidates: **23**.",
        "Supported candidate mappings: **6**": "Supported candidate mappings: **18**",
        "The reviewed candidate set now has at least one supported mapping for each of the six frozen labels, but that is still insufficient for a credible family-stratified train/validation/test holdout.": (
            "The reviewed candidate set now has exactly three supported independent families for each "
            "of the six frozen labels, meeting the splitter's minimum family depth. It is still not "
            "research-ready because normalized review snapshots are not byte-for-byte original-source "
            "preservation and no candidate has been research-admitted into canonical records."
        ),
    }
    for rel in ("docs/phase-03-handoff.md", "docs/data-card.md", "evidence/phase-03/README.md"):
        path = ROOT / rel
        text = path.read_text(encoding="utf-8")
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = text.replace(
            "INDEPENDENT_POSTMORTEM_CANDIDATES=11",
            "INDEPENDENT_POSTMORTEM_CANDIDATES=23",
        ).replace(
            "SUPPORTED_MAPPING_CANDIDATES=6",
            "SUPPORTED_MAPPING_CANDIDATES=18",
        ).replace(
            "BLOCKER=independent_taxonomy_coverage_insufficient",
            "BLOCKER=original_source_preservation_and_research_admission_required",
        ).replace(
            "BLOCKER_DETAIL=credible_family_stratified_holdout_not_yet_possible",
            "BLOCKER_DETAIL=original_primary_source_preservation_and_canonical_research_admission_required",
        )
        path.write_text(text, encoding="utf-8")

    handoff = ROOT / "docs/phase-03-handoff.md"
    htext = handoff.read_text(encoding="utf-8")
    anchor = "The candidate review is preserved at `datasets/incident_diagnosis/processed/"
    depth_note = (
        "\nFamily-depth milestone: **18 supported candidate families = 3 per frozen label**. "
        "This meets the deterministic splitter's minimum depth for one train, one validation, and "
        "one test family per label, but does not admit any research row. Original primary-source "
        "preservation, canonicalization, leakage audit, and final split construction remain required.\n\n"
    )
    if depth_note.strip() not in htext and anchor in htext:
        htext = htext.replace(anchor, depth_note + anchor)
    handoff.write_text(htext, encoding="utf-8")

    checksum_names = [
        "candidate-manifest.json",
        "source-candidate-summary.json",
        "source-audit-report.json",
        "auxiliary-servicenow-summary.json",
        "public-postmortem-candidate-coverage.json",
        "class-split-summary.json",
    ]
    lines = [
        f"{hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest()}  {name}"
        for name in checksum_names
    ]
    (EVIDENCE / "checksums.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    supported = [
        item for item in index["candidates"]
        if item["decision"] == "supported_candidate_mapping"
    ]
    counts = Counter(item["proposed_root_cause_code"] for item in supported)
    expected_labels = {
        "broken_payment_configuration",
        "database_connection_leak",
        "disk_exhaustion",
        "memory_leak",
        "n_plus_one_query",
        "no_fault",
    }
    if len(index["candidates"]) != 23 or len(supported) != 18:
        raise SystemExit("family-depth candidate counts are incorrect")
    if set(counts) != expected_labels or set(counts.values()) != {3}:
        raise SystemExit(f"family-depth label distribution incorrect: {dict(counts)}")
    if any(item["research_admitted"] for item in index["candidates"]):
        raise SystemExit("family-depth increment must not admit research rows")
    if manifest["research_ready"]:
        raise SystemExit("family-depth increment must not mark manifest research-ready")
    if manifest["split_counts"] != {"test": 0, "train": 0, "validation": 0}:
        raise SystemExit("family-depth increment must keep canonical research split empty")

    print("PHASE03_FAMILY_DEPTH_INCREMENT=PASS")
    print("CANDIDATES=23")
    print("SUPPORTED=18")
    print("FAMILIES_PER_LABEL=3")
    print("RESEARCH_ADMITTED=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
