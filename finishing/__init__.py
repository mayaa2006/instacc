"""Finishing toolkit: rendered takes + VideoJob -> phone-realistic final video + QA report.

Used by the agent (POST /api/posts/{id}/rendered with "takes") or by the video side directly:
    python\\python.exe -m finishing --job job.json --takes t1.mp4 t2.mp4 --out final.mp4
"""
from finishing.pipeline import FinishResult, finish

__all__ = ["finish", "FinishResult"]
