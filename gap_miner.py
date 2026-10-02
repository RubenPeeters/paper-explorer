#!/usr/bin/env python3
"""
gap_miner v1 - mine conference papers for research gaps.

Pipeline
  1. List papers per venue/year from DBLP.
  2. Enrich with abstract + open-access PDF link from OpenAlex (by DOI).
  3. Optionally download the PDF and keep the intro + the pages before the
     references (where limitations / future work / lessons learned live).
  4. Ask an LLM to fill a fixed schema per paper.
  5. Write out/papers.csv (one row per paper) and out/gaps.csv (one row per
     limitation / future-work / lesson / missing-resource statement).

Setup
  pip install requests anthropic pypdf
  export ANTHROPIC_API_KEY=...
  export CONTACT_EMAIL=you@university.be        # polite-pool for DBLP/OpenAlex
  export OPENALEX_API_KEY=...                   # only if OpenAlex asks for one

Usage
  python gap_miner.py --limit 10                # small trial run first
  python gap_miner.py --venues ISWC ESWC --years 2022-2026
  python gap_miner.py --list-only               # only test the DBLP listing

Runs are resumable: finished papers are cached in out/papers.jsonl.
"""

import argparse
import csv
import html
import io
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import requests

# --------------------------------------------------------------------------
# Configuration - edit this block
# --------------------------------------------------------------------------

# stream      : DBLP stream id of the conference series
# venue_regex : keeps main-conference volumes, drops co-located workshops
# keywords    : if set, keep only papers whose title matches one of them
#               (needed for large venues such as ACL)
TOPIC_KEYWORDS = [
    "knowledge graph", "knowledge base", "triple", "ontolog", "linked data",
    "relation extraction", "information extraction", "entity linking",
    "table", "tabular", "retrieval-augmented", "retrieval augmented", "rag",
    "graphrag", "sparql", "kbqa", "structured",
]

VENUES = {
    "ISWC":  {"stream": "conf/semweb", "venue_regex": r"^ISWC( \(\d+\))?$"},
    "ESWC":  {"stream": "conf/esws",   "venue_regex": r"^ESWC( \(\d+\))?$"},
    "EKAW":  {"stream": "conf/ekaw",   "venue_regex": r"^EKAW$"},
    "K-CAP": {"stream": "conf/kcap",   "venue_regex": r"^K-CAP$"},
    "ACL":   {"stream": "conf/acl",    "venue_regex": r"^ACL( \((\d+|Findings)\))?$",
              "keywords": TOPIC_KEYWORDS},
}

# Used by the LLM to rate how relevant each paper is to you.
INTERESTS = """\
- Knowledge graphs for cultural heritage, and for materials/adhesives.
- Extracting triples from text, and from text combined with tables.
- Using LLMs and knowledge graphs together (RAG, GraphRAG, KG-grounded QA)
  once the knowledge graph exists.
- Publishable outputs: research papers, resource papers (datasets,
  benchmarks, ontologies, tools) and in-use papers (real deployments)."""

MODEL = os.environ.get("GAP_MINER_MODEL", "claude-haiku-4-5-20251001")
CONTACT = os.environ.get("CONTACT_EMAIL", "")
OPENALEX_KEY = os.environ.get("OPENALEX_API_KEY", "")

DBLP_API = "https://dblp.org/search/publ/api"
# DBLP has used both spellings of the stream filter; the first that returns
# hits is used.
DBLP_QUERY_TEMPLATES = [
    "stream:streams/{stream}: year:{year}:",
    "streamid:{stream}: year:{year}:",
]
OPENALEX_API = "https://api.openalex.org/works"

HEAD_CHARS = 4000     # start of paper: abstract + introduction
TAIL_CHARS = 10000    # end of paper before references: discussion, conclusion
MAX_PDF_BYTES = 30 * 1024 * 1024

# --------------------------------------------------------------------------
# HTTP helper
# --------------------------------------------------------------------------

def make_session():
    s = requests.Session()
    ua = "gap-miner/1.0"
    if CONTACT:
        ua += f" (mailto:{CONTACT})"
    s.headers["User-Agent"] = ua
    return s


def get(session, url, params=None, tries=4, **kw):
    """GET with backoff on rate limits and server errors."""
    for attempt in range(tries):
        try:
            r = session.get(url, params=params, timeout=60, **kw)
        except requests.RequestException as e:
            if attempt == tries - 1:
                raise
            print(f"    network error ({e}); retrying", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code in (429, 500, 502, 503, 504) and attempt < tries - 1:
            wait = int(r.headers.get("Retry-After", 0)) or 10 * (attempt + 1)
            print(f"    HTTP {r.status_code}; waiting {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r
    raise RuntimeError("unreachable")

# --------------------------------------------------------------------------
# Step 1: list papers from DBLP
# --------------------------------------------------------------------------

def as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def extract_doi(info):
    if info.get("doi"):
        return info["doi"].lower()
    for ee in as_list(info.get("ee")):
        m = re.search(r"doi\.org/(10\.\S+)", ee)
        if m:
            return m.group(1).lower()
    return None


def dblp_hits(session, stream, year):
    for tmpl in DBLP_QUERY_TEMPLATES:
        q = tmpl.format(stream=stream, year=year)
        hits, offset = [], 0
        while True:
            r = get(session, DBLP_API,
                    params={"q": q, "format": "json", "h": 1000, "f": offset})
            if "json" not in r.headers.get("Content-Type", ""):
                sys.exit("DBLP returned a web page instead of JSON (most likely "
                         "its bot check). Try again later or from another network.")
            res = r.json()["result"]["hits"]
            batch = as_list(res.get("hit"))
            hits.extend(h["info"] for h in batch)
            offset += len(batch)
            if not batch or offset >= int(res.get("@total", 0)):
                break
            time.sleep(1)
        if hits:
            return hits
        time.sleep(1)
    return []


def list_papers(session, venue, year):
    cfg = VENUES[venue]
    rx = re.compile(cfg["venue_regex"])
    keywords = cfg.get("keywords")
    papers, dropped_venues, dropped_kw = [], Counter(), 0

    for info in dblp_hits(session, cfg["stream"], year):
        if str(info.get("year")) != str(year):
            continue
        if info.get("type") != "Conference and Workshop Papers":
            continue
        venue_strings = as_list(info.get("venue"))
        if not any(rx.match(v) for v in venue_strings):
            dropped_venues[" / ".join(venue_strings)] += 1
            continue
        title = html.unescape(info.get("title", "")).rstrip(".")
        if keywords and not any(k in title.lower() for k in keywords):
            dropped_kw += 1
            continue
        ees = as_list(info.get("ee"))
        papers.append({
            "id": info.get("key") or title,
            "venue": venue,
            "dblp_venue": " / ".join(venue_strings),
            "year": int(year),
            "title": title,
            "doi": extract_doi(info),
            "url": ees[0] if ees else info.get("url"),
        })

    msg = f"  {venue} {year}: kept {len(papers)}"
    if dropped_kw:
        msg += f", {dropped_kw} off-topic by title"
    if dropped_venues:
        top = ", ".join(f"{v} ({n})" for v, n in dropped_venues.most_common(4))
        msg += f", dropped other venues: {top}"
    print(msg)
    return papers

# --------------------------------------------------------------------------
# Step 2: enrich from OpenAlex
# --------------------------------------------------------------------------

def rebuild_abstract(inverted_index):
    if not inverted_index:
        return None
    positions = []
    for word, idxs in inverted_index.items():
        positions.extend((i, word) for i in idxs)
    return " ".join(w for _, w in sorted(positions))


def enrich(session, papers, batch_size=40):
    by_doi = {p["doi"]: p for p in papers if p.get("doi")}
    dois = list(by_doi)
    for i in range(0, len(dois), batch_size):
        chunk = dois[i:i + batch_size]
        params = {
            "filter": "doi:" + "|".join(chunk),
            "per-page": 100,
            "select": "doi,abstract_inverted_index,best_oa_location,open_access",
        }
        if CONTACT:
            params["mailto"] = CONTACT
        if OPENALEX_KEY:
            params["api_key"] = OPENALEX_KEY
        try:
            results = get(session, OPENALEX_API, params=params).json().get("results", [])
        except requests.HTTPError as e:
            print(f"  OpenAlex batch failed ({e}); continuing without it",
                  file=sys.stderr)
            continue
        for w in results:
            doi = (w.get("doi") or "").lower().replace("https://doi.org/", "")
            p = by_doi.get(doi)
            if not p:
                continue
            p["abstract"] = rebuild_abstract(w.get("abstract_inverted_index"))
            loc = w.get("best_oa_location") or {}
            p["pdf_url"] = loc.get("pdf_url") or (w.get("open_access") or {}).get("oa_url")
        time.sleep(0.3)

# --------------------------------------------------------------------------
# Step 3: full text (optional)
# --------------------------------------------------------------------------

def fetch_pdf_text(session, url):
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    try:
        r = get(session, url, tries=2, stream=True)
        data = r.raw.read(MAX_PDF_BYTES + 1, decode_content=True)
        if len(data) > MAX_PDF_BYTES or not data.startswith(b"%PDF"):
            return None
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        return None


def select_text(full):
    """Keep the start and the part just before the reference list."""
    refs = [m for m in re.finditer(r"\n\s*(References|REFERENCES|Bibliography)\s*\n", full)
            if m.start() > len(full) * 0.5]
    body = full[:refs[-1].start()] if refs else full
    if len(body) <= HEAD_CHARS + TAIL_CHARS:
        return body
    return body[:HEAD_CHARS] + "\n[... middle of paper omitted ...]\n" + body[-TAIL_CHARS:]


def paper_text(session, paper, use_fulltext):
    if use_fulltext and paper.get("pdf_url"):
        full = fetch_pdf_text(session, paper["pdf_url"])
        if full and len(full) > 3000:
            return select_text(full), "fulltext"
    if paper.get("abstract"):
        return paper["abstract"], "abstract"
    return None, "none"

# --------------------------------------------------------------------------
# Step 4: LLM extraction
# --------------------------------------------------------------------------

SYSTEM = (
    "You extract structured information from scientific papers for a "
    "research-gap analysis. Reply with one JSON object and nothing else. "
    "Use only what the text states. Use null or [] when the text does not "
    "say. Every quote must be copied verbatim from the text."
)

PROMPT = """\
The reader's research interests:
{interests}

Paper: "{title}" ({venue} {year})

Text ({source}):
<paper>
{text}
</paper>

Return a JSON object with exactly these keys:
- "contribution_type": "research" | "resource" | "in-use" | "other"
- "pipeline_stages": subset of ["ontology_or_schema", "kg_construction_from_text",
  "kg_construction_from_tables", "kg_construction_other", "kg_refinement",
  "kg_llm_integration", "querying_or_qa", "other"]
- "domain": application domain, or "general"
- "input_modality": e.g. "text", "tables", "text+tables", "images", "structured data"
- "problem": one sentence
- "method": one sentence
- "ontologies": list of ontology/vocabulary names used
- "eval_datasets": list of {{"name": str, "kind": "shared_benchmark" | "ad_hoc" | "unclear"}}
- "artifact_url": URL of released code/data if given, else null
- "gaps": list of {{"kind": "limitation" | "future_work" | "lesson_learned" | "missing_resource",
  "statement": one-sentence paraphrase, "quote": short verbatim supporting quote}}
- "relevance": integer 0-3, relevance to the reader's interests
- "relevance_reason": one short sentence
"""


def parse_json(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start:end + 1])


def extract(client, paper, text, source):
    prompt = PROMPT.format(interests=INTERESTS, title=paper["title"],
                           venue=paper["venue"], year=paper["year"],
                           source=source, text=text)
    last_error = None
    for _ in range(2):
        try:
            msg = client.messages.create(
                model=MODEL, max_tokens=2000, system=SYSTEM,
                messages=[{"role": "user", "content": prompt}])
            reply = "".join(b.text for b in msg.content if b.type == "text")
            return parse_json(reply)
        except Exception as e:           # bad JSON or API error: retry once
            last_error = e
            time.sleep(3)
    print(f"    extraction failed: {last_error}", file=sys.stderr)
    return None

# --------------------------------------------------------------------------
# Step 5: output
# --------------------------------------------------------------------------

def load_cache(path):
    done = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done[rec["id"]] = rec
    return done


def join(values):
    return " | ".join(str(v) for v in as_list(values) if v)


def write_csvs(records, out_dir):
    paper_cols = ["venue", "year", "title", "relevance", "relevance_reason",
                  "contribution_type", "pipeline_stages", "domain",
                  "input_modality", "problem", "method", "ontologies",
                  "eval_datasets", "artifact_url", "n_gaps", "text_source",
                  "doi", "url"]
    gap_cols = ["venue", "year", "title", "relevance", "contribution_type",
                "domain", "kind", "statement", "quote", "url"]

    def sort_key(r):
        rel = (r.get("extraction") or {}).get("relevance")
        return (-(rel if isinstance(rel, int) else -1), r["venue"], -r["year"])

    records = sorted(records, key=sort_key)
    with (out_dir / "papers.csv").open("w", newline="", encoding="utf-8") as fp, \
         (out_dir / "gaps.csv").open("w", newline="", encoding="utf-8") as fg:
        wp = csv.DictWriter(fp, fieldnames=paper_cols)
        wg = csv.DictWriter(fg, fieldnames=gap_cols)
        wp.writeheader()
        wg.writeheader()
        for r in records:
            x = r.get("extraction") or {}
            gaps = [g for g in as_list(x.get("gaps")) if isinstance(g, dict)]
            datasets = [f"{d.get('name')} [{d.get('kind')}]" if isinstance(d, dict) else str(d)
                        for d in as_list(x.get("eval_datasets"))]
            base = {"venue": r["venue"], "year": r["year"], "title": r["title"],
                    "relevance": x.get("relevance"),
                    "contribution_type": x.get("contribution_type"),
                    "domain": x.get("domain"), "url": r.get("url")}
            wp.writerow({**base,
                         "relevance_reason": x.get("relevance_reason"),
                         "pipeline_stages": join(x.get("pipeline_stages")),
                         "input_modality": x.get("input_modality"),
                         "problem": x.get("problem"), "method": x.get("method"),
                         "ontologies": join(x.get("ontologies")),
                         "eval_datasets": join(datasets),
                         "artifact_url": x.get("artifact_url"),
                         "n_gaps": len(gaps),
                         "text_source": r.get("text_source"),
                         "doi": r.get("doi")})
            for g in gaps:
                wg.writerow({**base, "kind": g.get("kind"),
                             "statement": g.get("statement"),
                             "quote": g.get("quote")})

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_years(spec):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in spec.split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--venues", nargs="+", default=list(VENUES), choices=list(VENUES))
    ap.add_argument("--years", default="2022-2026", help="e.g. 2022-2026 or 2023,2025")
    ap.add_argument("--out", default="out")
    ap.add_argument("--limit", type=int, help="process at most N new papers")
    ap.add_argument("--no-fulltext", action="store_true", help="abstracts only")
    ap.add_argument("--list-only", action="store_true",
                    help="list papers from DBLP and stop (no LLM calls)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "papers.jsonl"
    session = make_session()

    print("Listing papers from DBLP")
    papers = {}
    try:
        for venue in args.venues:
            for year in parse_years(args.years):
                for p in list_papers(session, venue, year):
                    papers[p["id"]] = p
                time.sleep(1)
    except requests.RequestException as e:
        sys.exit(f"Could not reach DBLP ({e}). Try again later or from another network.")
    print(f"Total: {len(papers)} papers")
    if not papers:
        sys.exit("Nothing found. Check DBLP_QUERY_TEMPLATES and the venue regexes.")
    if args.list_only:
        return

    done = load_cache(cache_path)
    todo = [p for pid, p in papers.items() if pid not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(done)} cached, {len(todo)} to process")

    if todo:
        import anthropic
        client = anthropic.Anthropic()
        print("Fetching abstracts and PDF links from OpenAlex")
        enrich(session, todo)

        with cache_path.open("a", encoding="utf-8") as cache:
            for n, p in enumerate(todo, 1):
                text, source = paper_text(session, p, not args.no_fulltext)
                p["text_source"] = source
                p["extraction"] = extract(client, p, text, source) if text else None
                p.pop("abstract", None)
                if text and p["extraction"] is None:
                    continue             # failed: leave uncached so it retries
                cache.write(json.dumps(p, ensure_ascii=False) + "\n")
                cache.flush()
                done[p["id"]] = p
                print(f"  [{n}/{len(todo)}] {source:8} {p['title'][:70]}")

    write_csvs(list(done.values()), out_dir)
    sources = Counter(r.get("text_source") for r in done.values())
    print(f"Wrote {out_dir/'papers.csv'} and {out_dir/'gaps.csv'}")
    print("Text used: " + ", ".join(f"{k}: {v}" for k, v in sources.items()))


if __name__ == "__main__":
    main()
