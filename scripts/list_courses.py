"""Aggregate docs_cache.json by course_id and print a summary.

Source of truth for what's in Qdrant when the server is offline.
"""
import json
import re
from collections import defaultdict
from pathlib import Path


def main() -> None:
    cache = Path("data/multimodal_db/docs_cache.json")
    data = json.loads(cache.read_text(encoding="utf-8"))

    by_course: dict[str, dict] = defaultdict(
        lambda: {
            "count": 0,
            "chapter_titles": set(),
            "domains": set(),
            "subjects": set(),
            "folders": set(),
            "files": set(),
        }
    )

    for d in data:
        m = d["metadata"]
        cid = m.get("course", "?")
        by_course[cid]["count"] += 1
        by_course[cid]["chapter_titles"].add(m.get("chapter_title", "?"))
        by_course[cid]["domains"].add(m.get("domain", "?"))

        src = m.get("source_file", "")
        # Normalise both `\` and `/` to `/` for path parsing
        parts = re.split(r"[\\/]+", src)
        if len(parts) >= 4:
            by_course[cid]["subjects"].add(parts[-3])
            by_course[cid]["folders"].add(parts[-2])
            by_course[cid]["files"].add(parts[-1])

    print(f"Total courses : {len(by_course)}")
    print(f"Total chunks  : {len(data)}")
    print()

    for cid, info in sorted(by_course.items(), key=lambda x: -x[1]["count"]):
        print(f"course_id   : {cid}")
        print(f"  chunks    : {info['count']}")
        print(f"  domain    : {', '.join(sorted(info['domains']))}")
        print(f"  subject   : {', '.join(sorted(info['subjects']))}")
        print(f"  folder    : {', '.join(sorted(info['folders']))}")
        print(f"  file      : {', '.join(sorted(info['files']))}")
        print(f"  chapters  : {', '.join(sorted(info['chapter_titles']))}")
        print()


if __name__ == "__main__":
    main()
