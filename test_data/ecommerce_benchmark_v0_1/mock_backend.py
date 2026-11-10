from __future__ import annotations
import json
from pathlib import Path

# Resource files are authoritative; mock_data.json is a migration backup.
_DATA_DIR = Path(__file__).parent / "resources"
if not _DATA_DIR.is_dir():
    raise FileNotFoundError(_DATA_DIR)
DATA = {p.stem: json.loads(p.read_text(encoding="utf-8-sig"))[p.stem]
        for p in sorted(_DATA_DIR.glob("*.json"))}

def _one(rows, **match):
    out = [r for r in rows if all(r.get(k) == v for k, v in match.items())]
    return {"records": out, "found": bool(out)}

def get_order_status(order_id: str):
    return _one(DATA["orders"], order_id=order_id)

def get_delivery_status(order_id: str):
    return _one(DATA["deliveries"], order_id=order_id)

def get_return_status(return_id_or_order_id: str):
    rows = [
        r for r in DATA["returns"]
        if r["return_id"] == return_id_or_order_id or r["order_id"] == return_id_or_order_id
    ]
    return {"records": rows, "found": bool(rows)}

def get_payment_status(order_id: str):
    return _one(DATA["payments"], order_id=order_id)

def get_customer_contact(customer_id: str):
    return _one(DATA["customer_contacts"], customer_id=customer_id)

def get_customer_profile(customer_id: str):
    return _one(DATA["customer_profiles"], customer_id=customer_id)
