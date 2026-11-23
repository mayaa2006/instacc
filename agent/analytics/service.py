"""Analytics: pull post metrics at checkpoints, tag memories with real results, and check whether
the judge's scores predict real performance (Spearman rank correlation)."""
from __future__ import annotations

import json
import logging
import time

log = logging.getLogger(__name__)


def engagement(platform: str, m: dict) -> tuple[float | None, float | None]:
    """(views, engagement rate) from raw platform metrics."""
    if platform == "instagram":
        views = m.get("views") or m.get("reach")
        inter = m.get("total_interactions")
        if inter is None:
            inter = sum((m.get(k) or 0) for k in ("likes", "comments", "shares", "saved"))
    else:
        views = m.get("view_count")
        inter = sum((m.get(k) or 0) for k in ("like_count", "comment_count", "share_count"))
    if not views:
        return (float(views) if views is not None else None), None
    return float(views), round(float(inter) / float(views), 4)


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 3) if den else None


class AnalyticsService:
    def __init__(self, settings, db, publishers: dict):
        self.s = settings
        self.db = db
        self.publishers = publishers

    def record(self, post_id: str, platform: str, checkpoint: str, data: dict) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO metrics(post_id,platform,checkpoint,collected_at,data) VALUES(?,?,?,?,?)",
            (post_id, platform, checkpoint, time.time(), json.dumps(data)),
        )

    def due_checkpoints(self, posts: list[dict]) -> list[tuple[dict, str, str, str]]:
        """(post, platform, external_id, checkpoint) pairs whose time has come and aren't collected."""
        have = {(r["post_id"], r["platform"], r["checkpoint"])
                for r in self.db.query("SELECT post_id, platform, checkpoint FROM metrics")}
        out = []
        now = time.time()
        for p in posts:
            for platform, info in (p.get("published") or {}).items():
                if not info or not info.get("id") or not info.get("at"):
                    continue
                for name, hours in self.s.agent.autopilot.metric_checkpoints_h.items():
                    if now >= info["at"] + hours * 3600 and (p["id"], platform, name) not in have:
                        out.append((p, platform, str(info["id"]), name))
        return out

    async def collect(self, posts: list[dict]) -> int:
        n = 0
        for p, platform, ext_id, cp in self.due_checkpoints(posts):
            client = self.publishers.get(platform)
            if client is None:
                continue
            try:
                if platform == "instagram":
                    data = await client.insights(ext_id)
                else:
                    data = (await client.video_stats([ext_id])).get(ext_id) or {}
            except Exception as e:
                log.warning("metrics %s/%s/%s failed: %s", p["id"], platform, cp, e)
                continue
            self.record(p["id"], platform, cp, data)
            n += 1
        if n:
            self.tag_memories()
        return n

    # --------------------------------------------------------------- report
    def table(self, checkpoint: str = "24h") -> list[dict]:
        rows = self.db.query(
            "SELECT m.post_id, m.platform, m.data, p.concept, p.idea_score, p.idea_task_id, p.caption_task_id, "
            "p.published FROM metrics m JOIN posts p ON p.id=m.post_id WHERE m.checkpoint=?", (checkpoint,))
        out = []
        for r in rows:
            data = json.loads(r["data"])
            views, er = engagement(r["platform"], data)
            url = (json.loads(r["published"] or "{}").get(r["platform"]) or {}).get("url")
            out.append({"post_id": r["post_id"], "platform": r["platform"], "concept": r["concept"],
                        "judge_score": r["idea_score"], "views": views, "engagement_rate": er, "url": url,
                        "idea_task_id": r["idea_task_id"], "caption_task_id": r["caption_task_id"], "raw": data})
        out.sort(key=lambda x: (x["views"] or 0), reverse=True)
        return out

    def calibration(self, checkpoint: str = "24h") -> dict:
        rows = [r for r in self.table(checkpoint) if r["judge_score"] is not None]
        res = {"checkpoint": checkpoint, "n": len(rows)}
        for metric in ("views", "engagement_rate"):
            pairs = [(r["judge_score"], r[metric]) for r in rows if r[metric] is not None]
            res[f"spearman_judge_vs_{metric}"] = spearman([a for a, _ in pairs], [b for _, b in pairs])
        rho = res.get("spearman_judge_vs_views")
        if res["n"] < 8:
            res["verdict"] = f"need more data ({res['n']}/8 posts with judge scores)"
        elif rho is None:
            res["verdict"] = "no variation to compare"
        elif rho >= 0.4:
            res["verdict"] = "judge predicts performance well: trust its rankings"
        elif rho >= 0.15:
            res["verdict"] = "weak signal: judge helps a little; keep rating evals"
        else:
            res["verdict"] = "judge does NOT predict performance: revise the rubric in config/tasks.yaml"
        return res

    def tag_memories(self, checkpoint: str = "24h") -> int:
        """Label approved memories with their real results so future prompts imitate what worked."""
        rows = [r for r in self.table(checkpoint) if r["views"] is not None]
        if len(rows) < 3:
            return 0
        views_sorted = sorted(r["views"] for r in rows)
        n = 0
        for r in rows:
            pct = sum(v <= r["views"] for v in views_sorted) / len(views_sorted)
            band = "top 20%" if pct >= 0.8 else "bottom 20%" if pct <= 0.2 else "average"
            er = f", ER {r['engagement_rate'] * 100:.1f}%" if r["engagement_rate"] is not None else ""
            tag = f"{band} ({int(r['views'])} views{er} on {r['platform']})"
            for tid in (r["idea_task_id"], r["caption_task_id"]):
                if not tid:
                    continue
                for m in self.db.query("SELECT id, meta FROM memory_items"):
                    meta = json.loads(m["meta"] or "{}")
                    if meta.get("task_id") == tid:
                        self.db.memory_update_meta(m["id"], {"perf": tag, "perf_views": r["views"]})
                        n += 1
        return n
