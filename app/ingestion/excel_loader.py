"""
Excel/CSV ingestion. Each sheet becomes one RawDocument whose content is a
readable, row-by-row rendering of the table (so it chunks and embeds
sensibly), with sheet_name/columns preserved in metadata.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.core.exceptions import IngestionError
from app.core.logging import get_logger
from app.rag.metadata import RawDocument, new_id

logger = get_logger(__name__)


def _dataframe_to_text(df: pd.DataFrame, max_rows: int = 500) -> str:
    df = df.head(max_rows)
    lines = [" | ".join(str(c) for c in df.columns)]
    for _, row in df.iterrows():
        lines.append(" | ".join(str(v) for v in row.values))
    return "\n".join(lines)


def load_excel(path: str | Path, source_type: str = "user_upload", title: str | None = None) -> list[RawDocument]:
    path = Path(path)
    if not path.exists():
        raise IngestionError(f"Spreadsheet not found: {path}")

    documents: list[RawDocument] = []
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            sheets = {"Sheet1": df}
        else:
            sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl" if path.suffix.lower() == ".xlsx" else None)
    except Exception as exc:  # noqa: BLE001
        raise IngestionError(f"Could not read spreadsheet: {exc}") from exc

    for sheet_name, df in sheets.items():
        if df.empty:
            continue
        text = _dataframe_to_text(df)
        documents.append(
            RawDocument(
                document_id=new_id("excel"),
                title=f"{title or path.stem} — {sheet_name}",
                content=text,
                source_type=source_type,
                source_url="",
                file_type="excel",
                category="user-provided-document",
                extra_metadata={"sheet_name": sheet_name, "columns": list(map(str, df.columns))},
            )
        )

    if not documents:
        raise IngestionError("No data found in the uploaded spreadsheet.")
    return documents
