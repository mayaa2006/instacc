"""SQLite storage. One file, WAL mode, safe to share across threads."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS persona_versions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at REAL NOT NULL,
  data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL,
  prompt TEXT NOT NULL,
  note TEXT,
  parent_id TEXT,
  source TEXT,
  status TEXT NOT NULL,
  result TEXT,
  chosen TEXT,
  error TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE TABLE IF NOT EXISTS trace_steps (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL,
  loop INTEGER NOT NULL DEFAULT 0,
  stage TEXT NOT NULL,
  slot TEXT,
  model TEXT,
  endpoint TEXT,
  role TEXT,
  temperature REAL,
  output TEXT,
  latency_ms INTEGER,
  cost_usd REAL,
  note TEXT,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trace_task ON trace_steps(task_id);
CREATE TABLE IF NOT EXISTS llm_usage (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  slot TEXT,
  endpoint TEXT,
  kind TEXT,
  model TEXT,
  prompt_tokens INTEGER,
  completion_tokens INTEGER,
  cost_usd REAL
);
CREATE TABLE IF NOT EXISTS memory_items (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,
  text TEXT NOT NULL,
  meta TEXT,
  embedding TEXT,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_kind ON memory_items(kind);
CREATE TABLE IF NOT EXISTS eval_runs (
  id TEXT PRIMARY KEY,
  name TEXT,
  config TEXT NOT NULL,
  status TEXT NOT NULL,
  progress TEXT,
  error TEXT,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS eval_items (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  case_id TEXT NOT NULL,
  task_type TEXT NOT NULL,
  prompt TEXT NOT NULL,
  variant TEXT NOT NULL,
  task_id TEXT,
  output TEXT,
  cost_usd REAL,
  wall_s REAL,
  error TEXT,
  UNIQUE(run_id, case_id, variant)
);
CREATE TABLE IF NOT EXISTS eval_votes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  case_id TEXT NOT NULL,
  a TEXT NOT NULL,
  b TEXT NOT NULL,
  winner TEXT NOT NULL,      -- variant name or 'tie'
  voter TEXT NOT NULL,       -- 'human' or 'llm:<slot>'
  reason TEXT,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_votes_run ON eval_votes(run_id);
CREATE TABLE IF NOT EXISTS series (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  status TEXT NOT NULL DEFAULT 'active',
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS continuity (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  series_id TEXT,                 -- NULL = applies to the whole persona
  kind TEXT NOT NULL,             -- outfit|location|prop|storyline|joke|fact|lighting|relationship
  key TEXT NOT NULL,
  value TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active',   -- proposed|active|retired
  source TEXT,
  last_used_post TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cont_series ON continuity(series_id, status);
CREATE TABLE IF NOT EXISTS posts (
  id TEXT PRIMARY KEY,
  series_id TEXT,
  status TEXT NOT NULL,
  concept TEXT NOT NULL,
  idea_task_id TEXT,
  idea_score REAL,
  platforms TEXT NOT NULL,
  scheduled_at REAL,
  video_job TEXT,
  media_path TEXT,
  caption TEXT,
  hashtags TEXT,
  caption_task_id TEXT,
  ai_label INTEGER NOT NULL DEFAULT 1,
  published TEXT,
  qa TEXT,
  error TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_posts_status ON posts(status);
CREATE TABLE IF NOT EXISTS comments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  platform TEXT NOT NULL,
  post_id TEXT,
  media_id TEXT,
  comment_id TEXT NOT NULL,
  parent_id TEXT,
  username TEXT,
  text TEXT NOT NULL,
  posted_at REAL,
  category TEXT,
  risk TEXT,
  draft TEXT,
  status TEXT NOT NULL DEFAULT 'new',
  reply_id TEXT,
  error TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  UNIQUE(platform, comment_id)
);
CREATE INDEX IF NOT EXISTS idx_comments_status ON comments(status);
CREATE TABLE IF NOT EXISTS metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  post_id TEXT NOT NULL,
  platform TEXT NOT NULL,
  checkpoint TEXT NOT NULL,       -- 1h|24h|7d|manual
  collected_at REAL NOT NULL,
  data TEXT NOT NULL,
  UNIQUE(post_id, platform, checkpoint)
);
CREATE TABLE IF NOT EXISTS kv (
  key TEXT PRIMARY KEY,
  value TEXT
);
"""


def _loads(v: str | None) -> Any:
    return json.loads(v) if v else None


class DB:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA)
            self._migrate()

    # Columns added after the first release. CREATE TABLE IF NOT EXISTS never alters an existing table,
    # so databases created by an older version get them here.
    _ADDED_COLUMNS = {"posts": {"qa": "TEXT"}}

    def _migrate(self) -> None:
        for table, cols in self._ADDED_COLUMNS.items():
            have = {r[1] for r in self._conn.execute(f"PRAGMA table_info({table})").fetchall()}
            for name, typ in cols.items():
                if name not in have:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {typ}")

    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def query(self, sql: str, params: tuple | dict = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def close(self) -> None:
        self._conn.close()

    # ---- kv -------------------------------------------------------------
    def kv_get(self, key: str, default: Any = None) -> Any:
        rows = self.query("SELECT value FROM kv WHERE key=?", (key,))
        return json.loads(rows[0]["value"]) if rows else default

    def kv_set(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )

    # ---- persona --------------------------------------------------------
    def persona_latest(self) -> dict | None:
        rows = self.query("SELECT id, created_at, data FROM persona_versions ORDER BY id DESC LIMIT 1")
        if not rows:
            return None
        return {"version": rows[0]["id"], "created_at": rows[0]["created_at"], "data": json.loads(rows[0]["data"])}

    def persona_save(self, data: dict) -> int:
        cur = self.execute(
            "INSERT INTO persona_versions(created_at, data) VALUES(?,?)", (time.time(), json.dumps(data))
        )
        return int(cur.lastrowid)

    def persona_history(self, limit: int = 20) -> list[dict]:
        return self.query(
            "SELECT id AS version, created_at FROM persona_versions ORDER BY id DESC LIMIT ?", (limit,)
        )

    # ---- tasks ----------------------------------------------------------
    def task_create(
        self, type_: str, prompt: str, *, note: str | None = None, parent_id: str | None = None,
        source: str | None = None,
    ) -> str:
        tid = uuid.uuid4().hex[:10]
        now = time.time()
        self.execute(
            "INSERT INTO tasks(id,type,prompt,note,parent_id,source,status,created_at,updated_at)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (tid, type_, prompt, note, parent_id, source, "queued", now, now),
        )
        return tid

    def task_update(self, tid: str, **fields: Any) -> None:
        if not fields:
            return
        for k in ("result", "chosen"):
            if k in fields and fields[k] is not None and not isinstance(fields[k], str):
                fields[k] = json.dumps(fields[k])
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        self.execute(f"UPDATE tasks SET {cols} WHERE id=?", (*fields.values(), tid))

    def _task_row(self, r: dict) -> dict:
        r["result"] = _loads(r.get("result"))
        r["chosen"] = _loads(r.get("chosen"))
        return r

    def task_get(self, tid: str) -> dict | None:
        rows = self.query("SELECT * FROM tasks WHERE id=?", (tid,))
        return self._task_row(rows[0]) if rows else None

    def task_list(self, status: str | None = None, limit: int = 50, include_eval: bool = False) -> list[dict]:
        where, params = [], []
        if status:
            where.append("status=?")
            params.append(status)
        if not include_eval:
            where.append("(source IS NULL OR source NOT LIKE 'eval:%')")
        sql = "SELECT * FROM tasks" + (" WHERE " + " AND ".join(where) if where else "")
        rows = self.query(sql + " ORDER BY created_at DESC LIMIT ?", (*params, limit))
        return [self._task_row(r) for r in rows]

    # ---- trace ----------------------------------------------------------
    def trace_add(self, task_id: str, stage: str, **f: Any) -> None:
        self.execute(
            "INSERT INTO trace_steps(task_id,loop,stage,slot,model,endpoint,role,temperature,output,"
            "latency_ms,cost_usd,note,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                task_id, f.get("loop", 0), stage, f.get("slot"), f.get("model"), f.get("endpoint"),
                f.get("role"), f.get("temperature"), f.get("output"), f.get("latency_ms"),
                f.get("cost_usd"), f.get("note"), time.time(),
            ),
        )

    def trace_get(self, task_id: str) -> list[dict]:
        return self.query("SELECT * FROM trace_steps WHERE task_id=? ORDER BY id", (task_id,))

    # ---- usage / budget -------------------------------------------------
    def usage_add(self, **f: Any) -> None:
        self.execute(
            "INSERT INTO llm_usage(ts,slot,endpoint,kind,model,prompt_tokens,completion_tokens,cost_usd)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (
                time.time(), f.get("slot"), f.get("endpoint"), f.get("kind"), f.get("model"),
                f.get("prompt_tokens", 0), f.get("completion_tokens", 0), f.get("cost_usd", 0.0),
            ),
        )

    def cloud_spend_this_month(self) -> float:
        now = datetime.now(timezone.utc)
        start = datetime(now.year, now.month, 1, tzinfo=timezone.utc).timestamp()
        rows = self.query(
            "SELECT COALESCE(SUM(cost_usd),0) AS s FROM llm_usage WHERE kind='cloud' AND ts>=?", (start,)
        )
        return float(rows[0]["s"])

    def memory_update_meta(self, item_id: int, patch: dict) -> None:
        rows = self.query("SELECT meta FROM memory_items WHERE id=?", (item_id,))
        if rows:
            meta = (json.loads(rows[0]["meta"]) if rows[0]["meta"] else {}) | patch
            self.execute("UPDATE memory_items SET meta=? WHERE id=?", (json.dumps(meta), item_id))

    # ---- memory ---------------------------------------------------------
    def memory_add(self, kind: str, text: str, meta: dict | None = None, embedding: list[float] | None = None) -> int:
        cur = self.execute(
            "INSERT INTO memory_items(kind,text,meta,embedding,created_at) VALUES(?,?,?,?,?)",
            (kind, text, json.dumps(meta or {}), json.dumps(embedding) if embedding else None, time.time()),
        )
        return int(cur.lastrowid)

    def memory_list(self, kind: str | None = None, limit: int = 500) -> list[dict]:
        if kind:
            rows = self.query(
                "SELECT * FROM memory_items WHERE kind=? ORDER BY id DESC LIMIT ?", (kind, limit)
            )
        else:
            rows = self.query("SELECT * FROM memory_items ORDER BY id DESC LIMIT ?", (limit,))
        for r in rows:
            r["meta"] = _loads(r["meta"]) or {}
            r["embedding"] = _loads(r["embedding"])
        return rows
