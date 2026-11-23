"""Telegram control: start tasks, get notified, approve / reject / rethink with one tap.

Only user IDs in TELEGRAM_ALLOWED_USERS can use the bot; everyone else is ignored.
"""
from __future__ import annotations

import html
import logging
import secrets
import time
from datetime import datetime
from pathlib import Path

from telegram import ForceReply, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters,
)

from agent.container import Container
from agent.content.posts import PostError
from agent.content.producer import summarize_job
from agent.evals.harness import EvalError
from agent.tasks import AUTO, TaskError

log = logging.getLogger(__name__)
MAX_MSG = 3900
MAX_VIDEO_BYTES = 49 * 1024 * 1024   # Bot API upload limit is 50 MB


def _when(ts: float | None) -> str:
    return datetime.fromtimestamp(ts).strftime("%a %d %b %H:%M") if ts else "unscheduled"


def _h(s) -> str:
    return html.escape(str(s or ""))


def format_task(task: dict) -> tuple[str, InlineKeyboardMarkup | None]:
    r = task.get("result") or {}
    head = f"🧠 <b>{_h(task['type'])}</b> · <code>{task['id']}</code>\n{_h(task['prompt'])}"
    if task.get("note"):
        head += f"\n<i>note: {_h(task['note'])}</i>"
    if task["status"] == "failed":
        return head + f"\n\n❌ failed: {_h(task.get('error'))}", None
    meta = []
    if r.get("best_score") is not None:
        meta.append(f"best {r['best_score']}/{r.get('threshold')}")
    if r.get("loops"):
        meta.append(f"{r['loops']} loop(s)")
    if r.get("below_threshold"):
        meta.append("⚠️ below threshold")
    body = [head, " · ".join(meta)] if meta else [head]
    for i, o in enumerate(r.get("options", []), 1):
        score = f"[{o['score']}] " if o.get("score") is not None else ""
        line = f"\n<b>{i}.</b> {score}{_h(o['text'])}"
        if o.get("rationale"):
            line += f"\n   <i>{_h(o['rationale'])}</i>"
        body.append(line)
    text = "\n".join(body)
    if len(text) > MAX_MSG:
        text = text[:MAX_MSG] + "…"
    kb = None
    if task["status"] == "awaiting_approval" and task["type"] == "plan":
        kb = plan_keyboard(task, set())
    elif task["status"] == "awaiting_approval":
        n = len(r.get("options", []))
        nums = [InlineKeyboardButton(f"✅ {i + 1}", callback_data=f"a:{task['id']}:{i}") for i in range(n)]
        rows = [nums[i:i + 5] for i in range(0, len(nums), 5)]
        rows.append([
            InlineKeyboardButton("🔁 Rethink", callback_data=f"t:{task['id']}"),
            InlineKeyboardButton("❌ Reject", callback_data=f"r:{task['id']}"),
            InlineKeyboardButton("📜 Trace", callback_data=f"x:{task['id']}"),
        ])
        kb = InlineKeyboardMarkup(rows)
    return text, kb


def plan_keyboard(task: dict, selected: set[int]) -> InlineKeyboardMarkup:
    n = len((task.get("result") or {}).get("options", []))
    nums = [InlineKeyboardButton(("☑️ " if i in selected else "☐ ") + str(i + 1), callback_data=f"pt:{task['id']}:{i}")
            for i in range(n)]
    rows = [nums[i:i + 4] for i in range(0, len(nums), 4)]
    rows.append([InlineKeyboardButton(f"📅 Schedule {len(selected)}" if selected else "📅 Schedule…",
                                      callback_data=f"ps:{task['id']}"),
                 InlineKeyboardButton("🔁 Rethink", callback_data=f"t:{task['id']}"),
                 InlineKeyboardButton("❌", callback_data=f"r:{task['id']}")])
    return InlineKeyboardMarkup(rows)


def format_trace(steps: list[dict]) -> str:
    lines = []
    total = 0.0
    for s in steps:
        total += s.get("cost_usd") or 0
        bits = [s["stage"], f"loop {s['loop']}", s.get("slot") or "?"]
        if s.get("role"):
            bits.append(s["role"])
        if s.get("latency_ms") is not None:
            bits.append(f"{s['latency_ms'] / 1000:.1f}s")
        if s.get("note"):
            bits.append(s["note"])
        lines.append("• " + " · ".join(_h(b) for b in bits))
    lines.append(f"\ncloud cost: ${total:.4f}")
    return "\n".join(lines)[:MAX_MSG]


class TelegramBot:
    def __init__(self, c: Container):
        self.c = c
        self.allowed = c.settings.secrets.telegram_allowed_users
        if not self.allowed:
            log.warning("TELEGRAM_ALLOWED_USERS is empty: the bot will ignore everyone")
        self.app = Application.builder().token(c.settings.secrets.telegram_token).build()
        self._pending_text: dict[str, str] = {}   # short key -> free text awaiting a task type
        self._rethink: dict[int, str] = {}         # force-reply message id -> task id
        self._replies: dict[int, tuple[str, str]] = {}   # force-reply message id -> (kind, id)
        self._plan_sel: dict[str, set[int]] = {}
        uf = filters.User(user_id=list(self.allowed)) if self.allowed else filters.User(user_id=[])
        types = list(c.settings.tasks.tasks)
        a = self.app
        a.add_handler(CommandHandler(["start", "help"], self.cmd_help, filters=uf))
        a.add_handler(CommandHandler("status", self.cmd_status, filters=uf))
        a.add_handler(CommandHandler("pending", self.cmd_pending, filters=uf))
        a.add_handler(CommandHandler("think", self.cmd_think, filters=uf))
        a.add_handler(CommandHandler("render", self.cmd_render, filters=uf))
        a.add_handler(CommandHandler("rate", self.cmd_rate, filters=uf))
        a.add_handler(CommandHandler("posts", self.cmd_posts, filters=uf))
        a.add_handler(CommandHandler("app", self.cmd_app, filters=uf))
        a.add_handler(CommandHandler("comments", self.cmd_comments, filters=uf))
        a.add_handler(CommandHandler([AUTO, *types], self.cmd_task, filters=uf))
        a.add_handler(CallbackQueryHandler(self.on_button))
        a.add_handler(MessageHandler(uf & filters.TEXT & ~filters.COMMAND, self.on_text))
        c.tasks.on_event(self.on_task_event)
        c.posts.on_event(self.on_post_event)
        c.comments.on_event(self.on_comment_event)

    async def start(self) -> None:
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)
        log.info("telegram bot polling")

    async def stop(self) -> None:
        try:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
        except Exception:
            log.exception("telegram shutdown")

    # ------------------------------------------------------------- outbound
    async def broadcast(self, text: str, kb=None) -> None:
        for uid in self.allowed:
            try:
                await self.app.bot.send_message(uid, text, parse_mode="HTML", reply_markup=kb)
            except Exception as e:
                log.warning("telegram send to %s failed: %s", uid, e)

    async def on_task_event(self, event: str, task: dict) -> None:
        if (task.get("source") or "").startswith(("post:", "eval:")):
            return  # post captions are sent with their video by on_post_event
        if event in ("ready", "failed"):
            text, kb = format_task(task)
            await self.broadcast(text, kb)

    async def send_video(self, path: str, caption: str) -> None:
        p = Path(path)
        if not p.exists() or p.stat().st_size > MAX_VIDEO_BYTES:
            await self.broadcast(caption + "\n<i>(video too large for Telegram; preview it in the dashboard)</i>")
            return
        for uid in self.allowed:
            try:
                with open(p, "rb") as f:
                    await self.app.bot.send_video(uid, f, caption=caption[:1000], parse_mode="HTML",
                                                  supports_streaming=True)
            except Exception as e:
                log.warning("send_video to %s failed: %s", uid, e)

    async def on_post_event(self, event: str, post: dict) -> None:
        pid = post["id"]
        head = f"<code>{pid}</code> · {_when(post.get('scheduled_at'))}\n{_h(post['concept'][:300])}"
        if event == "planned":
            await self.broadcast(f"🗓 planned · {head}")
        elif event == "script_ready":
            kb = InlineKeyboardMarkup([[InlineKeyboardButton("✅ Render", callback_data=f"pa:{pid}"),
                                        InlineKeyboardButton("🔁 Redo", callback_data=f"pr:{pid}"),
                                        InlineKeyboardButton("❌ Drop", callback_data=f"pd:{pid}")]])
            await self.broadcast(f"🎬 <b>shot list ready</b> · {head}\n\n{_h(summarize_job(post['video_job'] or {}))}"[:MAX_MSG], kb)
        elif event == "caption_review":
            task = self.c.db.task_get(post["caption_task_id"]) if post.get("caption_task_id") else None
            if post.get("media_path"):
                await self.send_video(post["media_path"], f"🎞 <b>rendered</b> · {head}")
            if task:
                text, kb = format_task(task)
                await self.broadcast(f"✍️ <b>pick a caption</b>: approving schedules the post for "
                                     f"{_when(post.get('scheduled_at'))}\n\n" + text, kb)
        elif event == "scheduled":
            kb = InlineKeyboardMarkup([[InlineKeyboardButton("🚀 Post now", callback_data=f"pn:{pid}"),
                                        InlineKeyboardButton("❌ Drop", callback_data=f"pd:{pid}")]])
            await self.broadcast(f"📅 <b>scheduled</b> · {head}", kb)
        elif event == "published":
            links = []
            for plat, info in (post.get("published") or {}).items():
                st = (info or {}).get("state")
                if st == "in_drafts":
                    links.append(f"{plat}: in your TikTok drafts: open the app, paste the caption, switch on the AI-generated label, post")
                elif info and info.get("url"):
                    links.append(f"{plat}: {info['url']}")
                elif st:
                    links.append(f"{plat}: {st}")
            err = f"\n⚠️ {_h(post['error'])}" if post.get("error") else ""
            await self.broadcast(f"✅ <b>published</b> · {head}\n" + "\n".join(_h(x) for x in links) + err)
        elif event == "failed":
            await self.broadcast(f"❌ <b>post failed</b> · {head}\n{_h(post.get('error'))}")
        elif event == "needs_reconnect":
            await self.broadcast(f"🔑 <b>{_h(post.get('platform'))} needs reconnecting</b>\n{_h(post.get('error'))}")

    async def on_comment_event(self, event: str, payload: dict) -> None:
        if event != "review":
            return
        for cm in payload["comments"][:15]:
            await self.broadcast(*self.comment_message(cm))

    def comment_message(self, cm: dict):
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("✅ Send", callback_data=f"cs:{cm['id']}"),
                                    InlineKeyboardButton("✏️ Edit", callback_data=f"ce:{cm['id']}"),
                                    InlineKeyboardButton("🙈 Ignore", callback_data=f"ci:{cm['id']}")]])
        risk = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(cm.get("risk"), "⚪")
        text = (f"💬 {risk} {_h(cm.get('category'))} · @{_h(cm.get('username'))}\n«{_h(cm['text'][:600])}»"
                f"\n\n↪️ {_h(cm.get('draft'))}")
        return text, kb

    # ------------------------------------------------------------- commands
    async def cmd_app(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """One-tap phone setup: a link that opens the app already signed in (single use, 10 min)."""
        pairing = self.c.pairing
        if pairing is None:
            await update.message.reply_text("The web app isn't running.")
            return
        code, _ = pairing.new_code()
        links = pairing.links(code)
        if not links:
            await update.message.reply_text("Your PC has no network address right now.")
            return
        kb = InlineKeyboardMarkup([[InlineKeyboardButton(f"📲 Open app ({l['label']})", url=l["url"])] for l in links])
        await update.message.reply_html(
            "Tap to open instacc on this phone, already signed in. The link works once, for 10 minutes.\n\n"
            "Then in Chrome: <b>⋮ → Add to Home screen</b> (or <b>Install app</b>) to get an icon.\n"
            "<i>Home Wi-Fi links only work when your phone is on the same Wi-Fi as the PC.</i>",
            reply_markup=kb)

    async def cmd_posts(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        posts = self.c.posts.list(active_only=True)
        if not posts:
            await update.message.reply_text("Nothing planned. Try /plan")
            return
        lines = [f"<code>{p['id']}</code> {_when(p.get('scheduled_at'))} · <b>{p['status'].replace('_', ' ')}</b>\n"
                 f"   {_h(p['concept'][:120])}" for p in posts[:25]]
        await update.message.reply_html("\n".join(lines)[:MAX_MSG])

    async def cmd_comments(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        pending = self.c.comments.list(status="drafted", limit=15)
        if not pending:
            await update.message.reply_text("No drafted replies waiting.")
        for cm in reversed(pending):
            text, kb = self.comment_message(cm)
            await update.message.reply_html(text, reply_markup=kb)

    async def cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        types = "\n".join(f"/{k} &lt;request&gt; — {_h(v.description)}" for k, v in self.c.settings.tasks.tasks.items())
        await update.message.reply_html(
            "<b>instacc</b>\n" + types +
            "\n/auto &lt;request&gt; — let the router pick"
            "\n/plan [focus] — a week of post ideas; tick the ones to schedule"
            "\n/posts — calendar · /comments — replies waiting"
            "\n/app — open the phone app (one tap, signed in)"
            "\n\n/pending — waiting for approval\n/status — GPU, budget\n/render [hours] · /think — GPU mode"
            "\n/rate [eval id] — blind-rate eval pairs"
            "\n\nOr just send text and pick a task type."
        )

    async def cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        g = self.c.gpu.status()
        spent = self.c.db.cloud_spend_this_month()
        pend = len(self.c.db.task_list(status="awaiting_approval"))
        running = len(self.c.db.task_list(status="running"))
        await update.message.reply_html(
            f"GPU: <b>{g['mode']}</b>{' (' + _h(g['holder']) + ')' if g['holder'] else ''}"
            f"\nwaiting for think mode: {g['calls_waiting_for_think_mode']}"
            f"\ntasks: {running} running · {pend} pending"
            f"\ncloud: ${spent:.2f} / ${self.c.settings.agent.budget.monthly_usd:.2f}"
        )

    async def cmd_pending(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        tasks = self.c.db.task_list(status="awaiting_approval", limit=10)
        if not tasks:
            await update.message.reply_text("Nothing waiting.")
        for t in reversed(tasks):
            text, kb = format_task(t)
            await update.message.reply_html(text, reply_markup=kb)

    async def cmd_think(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("Switching GPU to think mode…")
        await self.c.gpu.set_mode("think")
        await update.message.reply_text("GPU is in think mode.")

    async def cmd_render(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            hours = float(ctx.args[0]) if ctx.args else 2.0
        except ValueError:
            hours = 2.0
        await update.message.reply_text("Switching GPU to render mode…")
        await self.c.gpu.set_mode("render", holder="telegram", ttl_s=hours * 3600)
        await update.message.reply_text(f"GPU reserved for video for {hours:g}h.")

    async def cmd_task(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        cmd = update.message.text.split()[0].lstrip("/").split("@")[0]
        prompt = " ".join(ctx.args)
        if not prompt and cmd == "plan":
            prompt = "Plan next week's posts: a balanced mix across the content pillars, with at least one series episode."
        if not prompt:
            await update.message.reply_text(f"Usage: /{cmd} <what you want>")
            return
        await self._create(update, cmd, prompt)

    async def _create(self, update: Update, type_: str, prompt: str, note: str | None = None) -> None:
        try:
            tid = self.c.tasks.create(type_, prompt, note=note, source="telegram")
        except (TaskError, ValueError) as e:
            await update.effective_message.reply_text(f"Couldn't start: {e}")
            return
        extra = ""
        cfg = self.c.settings.tasks.tasks.get(type_)
        if self.c.gpu.mode == "render" and (cfg is None or cfg.sensitivity == "private"):
            extra = "\n(GPU is rendering; this private task will run when it returns to think mode.)"
        await update.effective_message.reply_text(f"🧠 Thinking ({type_}) · task {tid}{extra}")

    async def cmd_rate(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        rid = ctx.args[0] if ctx.args else None
        if not rid:
            runs = [r for r in self.c.evals.list_runs() if r["status"] == "done"]
            if not runs:
                await update.message.reply_text("No finished eval runs yet.")
                return
            rid = runs[0]["id"]
        await self._send_pair(update.effective_message, rid)

    async def _send_pair(self, message, rid: str) -> None:
        try:
            pair = self.c.evals.next_pair(rid)
        except EvalError as e:
            await message.reply_text(str(e))
            return
        if not pair:
            await message.reply_text(f"All pairs in eval {rid} rated. Report: /api/evals/{rid}")
            return
        tok = pair["token"]
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("⬅️ A", callback_data=f"v:{rid}:{tok}:left"),
            InlineKeyboardButton("🤝 tie", callback_data=f"v:{rid}:{tok}:tie"),
            InlineKeyboardButton("B ➡️", callback_data=f"v:{rid}:{tok}:right"),
        ]])
        text = (f"⚖️ <b>{_h(pair['task_type'])}</b>: {_h(pair['prompt'])}\n\n"
                f"<b>A</b>\n{_h(pair['left'])}\n\n<b>B</b>\n{_h(pair['right'])}\n\n<i>{pair['remaining']} left</i>")
        await message.reply_html(text[:MAX_MSG], reply_markup=kb)

    # --------------------------------------------------------------- inbound
    async def on_text(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        msg = update.message
        if msg.reply_to_message and msg.reply_to_message.message_id in self._replies:
            kind, ident = self._replies.pop(msg.reply_to_message.message_id)
            text = msg.text.strip()
            try:
                if kind == "comment":
                    cm = await self.c.comments.send(int(ident), text)
                    await msg.reply_text("✅ Sent" if cm["status"] == "sent" else f"Failed: {cm.get('error')}")
                elif kind == "script":
                    self.c.posts.redo_script(ident, None if text == "-" else text)
                    await msg.reply_text("🔁 Rewriting the shot list…")
            except (PostError, ValueError, KeyError) as e:
                await msg.reply_text(str(e))
            return
        if msg.reply_to_message and msg.reply_to_message.message_id in self._rethink:
            tid = self._rethink.pop(msg.reply_to_message.message_id)
            note = None if msg.text.strip() == "-" else msg.text.strip()
            try:
                new = self.c.tasks.rethink(tid, note, source="telegram")
                await msg.reply_text(f"🔁 Rethinking · task {new}")
            except TaskError as e:
                await msg.reply_text(str(e))
            return
        key = secrets.token_hex(4)
        self._pending_text[key] = msg.text
        if len(self._pending_text) > 50:
            self._pending_text.pop(next(iter(self._pending_text)))
        btns = [InlineKeyboardButton(k, callback_data=f"n:{key}:{k}") for k in [AUTO, *self.c.settings.tasks.tasks]]
        await msg.reply_text("What should I do with this?",
                             reply_markup=InlineKeyboardMarkup([btns[i:i + 3] for i in range(0, len(btns), 3)]))

    async def on_button(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        q = update.callback_query
        if q.from_user.id not in self.allowed:
            await q.answer()
            return
        kind, _, rest = (q.data or "").partition(":")
        try:
            if kind == "a":
                tid, _, idx = rest.partition(":")
                task = await self.c.tasks.approve(tid, int(idx))
                await q.answer("Approved")
                await q.edit_message_reply_markup(None)
                await q.message.reply_html(f"✅ Approved #{int(idx) + 1}:\n{_h(task['chosen']['text'])}")
            elif kind == "r":
                await self.c.tasks.reject(rest)
                await q.answer("Rejected")
                await q.edit_message_reply_markup(None)
            elif kind == "t":
                self.c.tasks._get_pending(rest)  # validate before asking for the note
                await q.answer()
                m = await q.message.reply_text(
                    "Reply with what to change (or '-' for no note):",
                    reply_markup=ForceReply(selective=True),
                )
                self._rethink[m.message_id] = rest
            elif kind == "pt":
                tid, _, idx = rest.partition(":")
                sel = self._plan_sel.setdefault(tid, set())
                sel ^= {int(idx)}
                await q.answer()
                task = self.c.db.task_get(tid)
                await q.edit_message_reply_markup(plan_keyboard(task, sel))
            elif kind == "ps":
                sel = self._plan_sel.get(rest) or set()
                if not sel:
                    await q.answer("Tap the numbers you want first", show_alert=True)
                    return
                await self.c.tasks.approve_many(rest, sorted(sel))
                self._plan_sel.pop(rest, None)
                await q.answer("Scheduled")
                await q.edit_message_reply_markup(None)
            elif kind in ("pa", "pr", "pd", "pn"):
                try:
                    if kind == "pa":
                        await self.c.posts.approve_script(rest)
                        await q.answer("Queued for render")
                        await q.edit_message_reply_markup(None)
                    elif kind == "pr":
                        await q.answer()
                        m = await q.message.reply_text("What should change in the shot list? ('-' for just redo)",
                                                       reply_markup=ForceReply(selective=True))
                        self._replies[m.message_id] = ("script", rest)
                    elif kind == "pd":
                        self.c.posts.drop(rest)
                        await q.answer("Dropped")
                        await q.edit_message_reply_markup(None)
                    elif kind == "pn":
                        self.c.posts._update(rest, scheduled_at=time.time())
                        await q.answer("Posting…")
                        await q.edit_message_reply_markup(None)
                        await self.c.posts.publish(rest)
                except PostError as e:
                    await q.answer(str(e)[:190], show_alert=True)
            elif kind in ("cs", "ce", "ci"):
                cid = int(rest)
                try:
                    if kind == "cs":
                        cm = await self.c.comments.send(cid)
                        await q.answer("Sent" if cm["status"] == "sent" else f"Failed: {cm.get('error')}"[:190])
                        await q.edit_message_reply_markup(None)
                    elif kind == "ci":
                        self.c.comments.ignore(cid)
                        await q.answer("Ignored")
                        await q.edit_message_reply_markup(None)
                    else:
                        await q.answer()
                        m = await q.message.reply_text("Reply with your version of the answer:",
                                                       reply_markup=ForceReply(selective=True))
                        self._replies[m.message_id] = ("comment", str(cid))
                except (ValueError, KeyError) as e:
                    await q.answer(str(e)[:190], show_alert=True)
            elif kind == "v":
                rid, _, rest2 = rest.partition(":")
                tok, _, choice = rest2.rpartition(":")
                try:
                    self.c.evals.vote(tok, choice)
                except EvalError as e:
                    await q.answer(str(e)[:190], show_alert=True)
                    return
                await q.answer("Recorded")
                await q.edit_message_reply_markup(None)
                await self._send_pair(q.message, rid)
            elif kind == "x":
                await q.answer()
                await q.message.reply_html(format_trace(self.c.db.trace_get(rest)))
            elif kind == "n":
                key, _, type_ = rest.partition(":")
                text = self._pending_text.pop(key, None)
                await q.answer()
                if text is None:
                    await q.message.reply_text("That message expired; send it again.")
                    return
                await q.edit_message_reply_markup(None)
                await self._create(update, type_, text)
            else:
                await q.answer()
        except TaskError as e:
            await q.answer(str(e)[:190], show_alert=True)
