"""Load the evaluation dataset (evals/dataset.yaml) and derive what can be derived."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from t2sql.config import Settings
from t2sql.sql_guard import validate_sql

DATASET_PATH = Path(__file__).with_name("dataset.yaml")
LEVELS = ("easy", "medium", "hard", "out_of_scope")


@dataclass(frozen=True)
class Item:
    id: str
    level: str
    language: str
    question: str
    sql: str | None
    tables: tuple[str, ...]
    order_matters: bool = False

    @property
    def in_scope(self) -> bool:
        return self.level != "out_of_scope"


def load_dataset(path: Path | str = DATASET_PATH) -> list[Item]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    items = [
        Item(
            id=r["id"],
            level=r["level"],
            language=r["language"],
            question=r["question"],
            sql=r.get("sql"),
            tables=tuple(r.get("tables") or ()),
            order_matters=bool(r.get("order_matters", False)),
        )
        for r in raw
    ]
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ids in the dataset")
    return items


def reference_tables(sql: str, settings: Settings) -> frozenset[str]:
    """Tables a reference query reads, as seen by the same guard the agent uses (no CTE names)."""
    return validate_sql(sql, set(settings.allowed_tables), settings.max_rows).tables
