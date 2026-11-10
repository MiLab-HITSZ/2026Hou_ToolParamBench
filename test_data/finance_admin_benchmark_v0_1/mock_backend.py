import json
from pathlib import Path
# Resource files are authoritative; mock_data.json is a migration backup.
_DATA_DIR = Path(__file__).parent / "resources"
if not _DATA_DIR.is_dir():
    raise FileNotFoundError(_DATA_DIR)
DATA = {p.stem: json.loads(p.read_text(encoding="utf-8-sig"))[p.stem]
        for p in sorted(_DATA_DIR.glob("*.json"))}

def lookup(table, **match):
    rows=[x for x in DATA[table] if all(x.get(k)==v for k,v in match.items())]
    return {"records":rows,"found":bool(rows)}
