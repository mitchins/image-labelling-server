"""Lossless exports shared by the HTTP API and CLI."""

import json
import sqlite3

from config import decode_metadata_value, quote_identifier, validate_metadata_fields


def _settings(conn: sqlite3.Connection) -> dict:
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='settings'"
    ).fetchone() is None:
        return {}
    values = {}
    for key, raw_value in conn.execute("SELECT key, value FROM settings"):
        try:
            values[key] = json.loads(raw_value)
        except (json.JSONDecodeError, TypeError):
            values[key] = raw_value
    return values


def build_export_payload(conn: sqlite3.Connection, exclude_refuse: bool = False, config=None):
    """Build the mode-appropriate JSON export without deriving training labels."""
    conn.row_factory = sqlite3.Row
    settings = _settings(conn)
    if config is not None:
        settings.update({
            "mode": getattr(config, "mode", settings.get("mode", "classification")),
            "ontology_id": getattr(config, "ontology_id", settings.get("ontology_id")),
            "ontology_version": getattr(config, "ontology_version", settings.get("ontology_version")),
            "ontology": getattr(config, "ontology", settings.get("ontology", [])),
            "metadata_fields": getattr(config, "metadata_fields", settings.get("metadata_fields", [])),
        })
    if settings.get("mode") == "ontology_confirmation":
        metadata_fields = validate_metadata_fields(settings.get("metadata_fields") or [])
        columns = [
            "id", "path", "media_type", "cluster_id", "predicted_style",
            "predicted_confidence", "indicative_value", "confirmation",
            "confirmation_at", "confirmation_session_id",
        ] + metadata_fields
        select_columns = ", ".join(quote_identifier(column) for column in columns)
        rows = conn.execute(
            f"SELECT {select_columns} FROM queue "
            "WHERE confirmation IS NOT NULL ORDER BY confirmation_at, id"
        ).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            for field in metadata_fields:
                item[field] = decode_metadata_value(item[field])
            items.append(item)
        return {
            "mode": "ontology_confirmation",
            "ontology": {
                "id": settings.get("ontology_id"),
                "version": settings.get("ontology_version"),
                "values": settings.get("ontology") or [],
            },
            "items": items,
        }

    query = """
        SELECT path, media_type, human_label, cluster_id, predicted_style, labeled_at
        FROM queue WHERE human_label IS NOT NULL
    """
    if exclude_refuse:
        query += " AND human_label != 'REFUSE'"
    query += " ORDER BY labeled_at"
    return [
        {
            "path": row["path"],
            "media_type": row["media_type"],
            "label": row["human_label"],
            "cluster_id": row["cluster_id"],
            "predicted_style": row["predicted_style"],
            "labeled_at": row["labeled_at"],
        }
        for row in conn.execute(query).fetchall()
    ]
