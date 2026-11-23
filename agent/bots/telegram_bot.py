"""Telegram control: start tasks, get notified, approve / reject / rethink with one tap.

Only user IDs in TELEGRAM_ALLOWED_USERS can use the bot; everyone else is ignored.
"""
from __future__ import annotations

import html
import logging
import secrets

from telegram import ForceReply, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters,
)

from agent.container import Container
from agent.tasks import TaskError

log = logging.getLogger(__name__)
MAX_MSG = 3900


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
    if task["status"] == "awaiting_approval":
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
        uf = filters.User(user_id=list(self.allowed)) if self.allowed else filters.User(user_id=[])
        types = list(c.settings.tasks.tasks)
        a = self.app
        a.add_handler(CommandHandler(["start", "help"], self.cmd_help, filters=uf))
        a.add_handler(CommandHandler("status", self.cmd_status, filters=uf))
        a.add_handler(CommandHandler("pending", self.cmd_pending, filters=uf))
        a.add_handler(CommandHandler("think", self.cmd_think, filters=uf))
        a.add_handler(CommandHandler("render", self.cmd_render, filters=uf))
        a.add_handler(CommandHandler(types, self.cmd_task, filters=uf))
        a.add_handler(CallbackQueryHandler(self.on_button))
        a.add_handler(MessageHandler(uf & filters.TEXT & ~filters.COMMAND, self.on_text))
        c.tasks.on_event(self.on_task_event)

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
        if event in ("ready", "failed"):
            text, kb = format_task(task)
            await self.broadcast(text, kb)

    # ------------------------------------------------------------- commands
    async def cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        types = "\n".join(f"/{k} &lt;request&gt; — {_h(v.description)}" for k, v in self.c.settings.tasks.tasks.items())
        await update.message.reply_html(
            "<b>instacc</b>\n" + types +
            "\n\n/pending — waiting for approval\n/status — GPU, budget\n/render [hours] · /think — GPU mode"
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
        if self.c.gpu.mode == "render" and self.c.settings.tasks.tasks[type_].sensitivity == "private":
            extra = "\n(GPU is rendering; this private task will run when it returns to think mode.)"
        await update.effective_message.reply_text(f"🧠 Thinking ({type_}) · task {tid}{extra}")

    # --------------------------------------------------------------- inbound
    async def on_text(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        msg = update.message
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
        btns = [InlineKeyboardButton(k, callback_data=f"n:{key}:{k}") for k in self.c.settings.tasks.tasks]
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
