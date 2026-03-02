import json
from collections import defaultdict
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api.provider import ProviderRequest
from astrbot.api import logger, AstrBotConfig


# ── 多格式适配工具函数 ──────────────────────────────────────────────

def _norm_role(ctx: dict) -> str:
    """统一角色名：Gemini 用 'model'，其余用 'assistant'"""
    r = ctx.get("role", "")
    return "assistant" if r == "model" else r


def _get_text(ctx: dict) -> str:
    """从任意格式的上下文中提取文本内容"""
    content = ctx.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text" and p.get("text"))
    # Gemini parts
    parts = ctx.get("parts")
    if isinstance(parts, list):
        return " ".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text"))
    return ""


def _get_tool_calls(ctx: dict, skip_name: str = "") -> list[dict]:
    """从任意格式的 assistant/model 消息中提取工具调用。
    返回 [{id, name, arguments}]"""
    result = []
    # OpenAI / OpenAI-Responses 格式
    for tc in ctx.get("tool_calls", []):
        func = tc.get("function", {})
        name = func.get("name", "unknown")
        if skip_name and name == skip_name:
            continue
        result.append({
            "id": tc.get("id", ""),
            "name": name,
            "arguments": func.get("arguments", ""),
        })
    # Anthropic / Claude 格式：content 列表中的 tool_use
    content = ctx.get("content")
    if isinstance(content, list):
        for p in content:
            if isinstance(p, dict) and p.get("type") == "tool_use":
                name = p.get("name", "unknown")
                if skip_name and name == skip_name:
                    continue
                inp = p.get("input", {})
                result.append({
                    "id": p.get("id", ""),
                    "name": name,
                    "arguments": json.dumps(inp, ensure_ascii=False) if isinstance(inp, dict) else str(inp),
                })
    # Gemini 格式：parts 中的 functionCall（无原生 ID，按名称计数合成唯一 ID）
    parts = ctx.get("parts")
    if isinstance(parts, list):
        name_counts: dict[str, int] = {}
        for p in parts:
            if isinstance(p, dict) and "functionCall" in p:
                fc = p["functionCall"]
                name = fc.get("name", "unknown")
                if skip_name and name == skip_name:
                    continue
                args = fc.get("args", {})
                n = name_counts.get(name, 0)
                name_counts[name] = n + 1
                result.append({
                    "id": f"__gemini__{name}__{n}",
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args),
                })
    return result


def _get_tool_results(ctx: dict) -> list[dict]:
    """从任意格式的消息中提取工具调用结果。
    返回 [{call_id, content}]"""
    result = []
    # OpenAI 格式：role=tool 的独立消息
    if ctx.get("role") == "tool":
        result.append({
            "call_id": ctx.get("tool_call_id", ""),
            "content": ctx.get("content", ""),
        })
        return result
    # Anthropic 格式：user role，content 中有 tool_result
    content = ctx.get("content")
    if isinstance(content, list):
        for p in content:
            if isinstance(p, dict) and p.get("type") == "tool_result":
                res = p.get("content", "")
                if isinstance(res, list):
                    res = " ".join(part.get("text", "") for part in res if isinstance(part, dict))
                result.append({
                    "call_id": p.get("tool_use_id", ""),
                    "content": res,
                })
    # Gemini 格式：user role，parts 中有 functionResponse（按名称计数，与 _get_tool_calls 保持一致）
    parts = ctx.get("parts")
    if isinstance(parts, list):
        name_counts: dict[str, int] = {}
        for p in parts:
            if isinstance(p, dict) and "functionResponse" in p:
                fr = p["functionResponse"]
                name = fr.get("name", "")
                resp = fr.get("response", {})
                n = name_counts.get(name, 0)
                name_counts[name] = n + 1
                result.append({
                    "call_id": f"__gemini__{name}__{n}",
                    "content": json.dumps(resp, ensure_ascii=False) if isinstance(resp, dict) else str(resp),
                })
    return result


def _remove_tool_calls(ctx: dict, keep_ids: set) -> dict | None:
    """从上下文中移除不在 keep_ids 中的工具调用/结果。
    返回修改后的 ctx，或 None 表示整条消息应被丢弃。"""
    role = ctx.get("role", "")
    content = ctx.get("content")
    parts = ctx.get("parts")

    # 预提取 Gemini keep 名称（支持 __gemini__<name>__<n> 和旧格式 __gemini__<name>）
    _keep_gemini_names: set[str] = set()
    for kid in keep_ids:
        if kid.startswith("__gemini__"):
            tail = kid[len("__gemini__"):]
            idx = tail.rfind("__")
            _keep_gemini_names.add(tail[:idx] if idx > 0 and tail[idx + 2:].isdigit() else tail)

    # OpenAI role=tool 消息
    if role == "tool":
        return ctx if ctx.get("tool_call_id") in keep_ids else None

    # OpenAI/OpenAI-Responses assistant with tool_calls
    if ctx.get("tool_calls"):
        kept = [tc for tc in ctx["tool_calls"] if tc.get("id") in keep_ids]
        has_text = bool(ctx.get("content"))
        if not kept and not has_text:
            return None
        new_ctx = dict(ctx)
        if kept:
            new_ctx["tool_calls"] = kept
        else:
            del new_ctx["tool_calls"]
        return new_ctx

    # Anthropic assistant with tool_use in content
    if role == "assistant" and isinstance(content, list):
        if any(isinstance(p, dict) and p.get("type") == "tool_use" for p in content):
            new_content = [
                p for p in content
                if not (isinstance(p, dict) and p.get("type") == "tool_use") or p.get("id") in keep_ids
            ]
            if not new_content:
                return None
            return {**ctx, "content": new_content}

    # Anthropic user with tool_result in content
    if role == "user" and isinstance(content, list):
        if any(isinstance(p, dict) and p.get("type") == "tool_result" for p in content):
            new_content = [
                p for p in content
                if not (isinstance(p, dict) and p.get("type") == "tool_result") or p.get("tool_use_id") in keep_ids
            ]
            if not new_content:
                return None
            return {**ctx, "content": new_content}

    # Gemini model with functionCall in parts
    if role == "model" and isinstance(parts, list):
        if any(isinstance(p, dict) and "functionCall" in p for p in parts):
            new_parts = [
                p for p in parts
                if not (isinstance(p, dict) and "functionCall" in p)
                or p['functionCall'].get('name', '') in _keep_gemini_names
            ]
            if not new_parts:
                return None
            return {**ctx, "parts": new_parts}

    # Gemini user with functionResponse in parts
    if role == "user" and isinstance(parts, list):
        if any(isinstance(p, dict) and "functionResponse" in p for p in parts):
            new_parts = [
                p for p in parts
                if not (isinstance(p, dict) and "functionResponse" in p)
                or p['functionResponse'].get('name', '') in _keep_gemini_names
            ]
            if not new_parts:
                return None
            return {**ctx, "parts": new_parts}

    # 丢弃空的 assistant/model 消息
    if role in ("assistant", "model") and not content and not parts:
        return None

    return ctx


# ── 插件主体 ────────────────────────────────────────────────────────

@register("tool_use_compacter", "Lijinhao999", "参考自astrbot_plugin_tool_use_cleaner，清除历史工具调用记录以减少token浪费，并提供LLM可查询的回溯能力。支持缓存调用记录轮数控制、重置关键词检测。", "2.1.0")
class ToolUseCleanerPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.tool_context_keep_rounds = int(self.config.get("tool_context_keep_rounds", 0))
        self.compressed_keep_rounds = int(self.config.get("compressed_record_keep_rounds", 0))
        self.compressed_records: dict[str, list[dict]] = defaultdict(list)
        self.round_counter: dict[str, int] = defaultdict(int)
        keep_str = "全部清除" if self.tool_context_keep_rounds <= 0 else f"保留最近{self.tool_context_keep_rounds}轮"
        cache_str = "不限" if self.compressed_keep_rounds <= 0 else f"最近{self.compressed_keep_rounds}轮"
        logger.info(f"[压缩器] 已初始化：上下文工具调用{keep_str}，压缩缓存保留{cache_str}")

    def _collect_self_tool_ids(self, contexts: list[dict]) -> set:
        """收集自身工具 query_compressed_tools 的调用 ID"""
        ids = set()
        for ctx in contexts:
            for call in _get_tool_calls(ctx):   # 不跳过，捞全部
                if call["name"] == "query_compressed_tools":
                    ids.add(call["id"])
        return ids

    def _find_round_ends(self, contexts: list[dict]) -> list[int]:
        """找到所有带文本回复的完整轮次结束索引（适配多格式）"""
        def is_text_assistant(ctx: dict) -> bool:
            if _norm_role(ctx) != "assistant":
                return False
            return bool(_get_text(ctx))

        round_ends = []
        for i in range(len(contexts) - 1):
            if is_text_assistant(contexts[i]) and _norm_role(contexts[i + 1]) in ("user", "system"):
                round_ends.append(i)
        if contexts and is_text_assistant(contexts[-1]):
            round_ends.append(len(contexts) - 1)
        return round_ends

    def _extract_tool_records(self, contexts: list[dict], start: int, end: int, round_num: int) -> list[dict]:
        """从指定范围提取工具调用记录（支持 OpenAI / Anthropic / Gemini）。
        使用列表 + 首条未匹配策略，避免 ID 碰撞（适用于 Gemini 合成 ID、空 ID 等情况）。"""
        tool_calls_list: list[dict] = []
        last_user_msg = ""

        def _match_result(call_id: str, content: str):
            """将结果写入第一条 call_id 相同且尚无结果的调用记录"""
            for record in tool_calls_list:
                if record["tool_call_id"] == call_id and record["result"] is None:
                    record["result"] = content
                    return

        for i in range(start, min(end + 1, len(contexts))):
            ctx = contexts[i]
            role = _norm_role(ctx)

            if role == "user":
                text = _get_text(ctx)
                if text:
                    last_user_msg = text
                # Anthropic / Gemini 结果在 user 消息中
                for res in _get_tool_results(ctx):
                    _match_result(res["call_id"], res["content"])

            elif role == "assistant":
                for call in _get_tool_calls(ctx, skip_name="query_compressed_tools"):
                    tool_calls_list.append({
                        "tool_name": call["name"],
                        "tool_call_id": call["id"],
                        "arguments": call["arguments"],
                        "result": None,
                        "user_query": last_user_msg,
                        "round_num": round_num,
                    })

            # OpenAI role=tool 结果（_get_tool_results 对非 tool role 返回空列表，不影响其他格式）
            for res in _get_tool_results(ctx):
                _match_result(res["call_id"], res["content"])

        return tool_calls_list

    def _clean_contexts(self, contexts: list[dict], keep_ids: set) -> list[dict]:
        """对上下文列表执行工具调用清理"""
        cleaned = []
        for ctx in contexts:
            new_ctx = _remove_tool_calls(ctx, keep_ids)
            if new_ctx is not None:
                cleaned.append(new_ctx)
        return cleaned

    def _store_records(self, session_id: str, records: list[dict]) -> int:
        """存储压缩记录，返回实际新增的条数"""
        if not records:
            return 0
        existing = {r["tool_call_id"] for r in self.compressed_records[session_id]}
        new_records = [r for r in records if r["tool_call_id"] not in existing]
        if not new_records:
            return 0
        self.compressed_records[session_id].extend(new_records)
        return len(new_records)

    def _trim_records_by_rounds(self, session_id: str, current_round: int):
        """按轮数淘汰过旧的缓存记录。
        0 = 本轮即淘汰（保留0轮），N > 0 = 保留最近 N 轮"""
        if not self.compressed_records[session_id]:
            return
        # cutoff = current_round - K：round_num > cutoff 的记录保留
        # K=0 时 cutoff=current_round，当前轮存入的记录（round_num==current_round）立即被淘汰
        cutoff = current_round - self.compressed_keep_rounds
        before = len(self.compressed_records[session_id])
        self.compressed_records[session_id] = [
            r for r in self.compressed_records[session_id] if r["round_num"] > cutoff
        ]
        trimmed = before - len(self.compressed_records[session_id])
        if trimmed > 0:
            logger.debug(f"[压缩器] 淘汰 {trimmed} 条过期缓存记录（超过 {self.compressed_keep_rounds} 轮），剩余 {len(self.compressed_records[session_id])} 条")

    @filter.on_llm_request(priority=-1)
    async def clean_context(self, event: AstrMessageEvent, req: ProviderRequest):
        """压缩上下文中的工具调用：提取记录到缓存后从请求体中移除"""
        session_id = event.unified_msg_origin
        current_round = self.round_counter[session_id]
        self.round_counter[session_id] += 1

        ctx_count = len(req.contexts) if req.contexts else 0
        logger.info(f"[压缩器] 第{current_round}轮 | filter 触发，上下文共 {ctx_count} 条")

        if not req.contexts:
            return

        try:
            original_count = len(req.contexts)

            # 诊断日志：记录格式信息
            roles = [ctx.get("role", "?") for ctx in req.contexts]
            has_openai = any(ctx.get("tool_calls") for ctx in req.contexts)
            has_anthropic = any(
                isinstance(ctx.get("content"), list) and
                any(isinstance(p, dict) and p.get("type") in ("tool_use", "tool_result") for p in ctx["content"])
                for ctx in req.contexts
            )
            has_gemini = any(
                isinstance(ctx.get("parts"), list) and
                any(isinstance(p, dict) and ("functionCall" in p or "functionResponse" in p) for p in ctx["parts"])
                for ctx in req.contexts
            )
            has_tool_role = any(ctx.get("role") == "tool" for ctx in req.contexts)
            logger.info(
                f"[压缩器] 第{current_round}轮 上下文{original_count}条 "
                f"角色={roles[:12]} "
                f"格式: openai={has_openai} anthropic={has_anthropic} gemini={has_gemini} tool_role={has_tool_role}"
            )

            self_tool_ids = self._collect_self_tool_ids(req.contexts)
            newly_cached = 0

            if self.tool_context_keep_rounds <= 0:
                # 全部清除模式
                records = self._extract_tool_records(req.contexts, 0, len(req.contexts) - 1, current_round)
                newly_cached = self._store_records(session_id, records)
                req.contexts = self._clean_contexts(req.contexts, self_tool_ids)
            else:
                # 保留N轮模式
                round_ends = self._find_round_ends(req.contexts)
                cutoff_index = -1
                if round_ends:
                    if len(round_ends) <= self.tool_context_keep_rounds:
                        logger.debug(f"[压缩器] 共{len(round_ends)}轮 ≤ 保留{self.tool_context_keep_rounds}轮，跳过清理")
                    else:
                        cutoff_index = round_ends[-(self.tool_context_keep_rounds + 1)]
                        logger.debug(f"[压缩器] 共{len(round_ends)}轮，保留{self.tool_context_keep_rounds}轮，清理到索引{cutoff_index}")

                if cutoff_index >= 0:
                    records = self._extract_tool_records(req.contexts, 0, cutoff_index, current_round)
                    newly_cached = self._store_records(session_id, records)
                    pre = req.contexts[:cutoff_index + 1]
                    post = req.contexts[cutoff_index + 1:]
                    req.contexts = self._clean_contexts(pre, self_tool_ids) + post

            self._trim_records_by_rounds(session_id, current_round)

            removed_count = original_count - len(req.contexts)
            if removed_count > 0:
                cached_total = len(self.compressed_records.get(session_id, []))
                cached_names = ", ".join(sorted({r["tool_name"] for r in self.compressed_records.get(session_id, [])})) or "无"
                expire_str = "不保留（0轮，立即淘汰）" if self.compressed_keep_rounds == 0 else f"{self.compressed_keep_rounds}轮后过期"
                logger.info(
                    f"[压缩器] 第{current_round}轮 | "
                    f"已清除 {removed_count} 条 tool_call/result 消息并缓存 | "
                    f"历史记录共 {cached_total} 条（{cached_names}），{expire_str}"
                )
            else:
                logger.debug(f"[压缩器] 第{current_round}轮 | 上下文无工具调用消息，跳过")
        except Exception as e:
            logger.error(f"[压缩器] 第{current_round}轮处理出错：{e}", exc_info=True)

    @filter.llm_tool(name="query_compressed_tools")
    async def query_compressed_tools(self, event: AstrMessageEvent, tool_name: str = ""):
        '''你在会话中的历史调用工具记录将会被压缩删除。此工具用于查询当前会话中被压缩的历史工具调用记录。当你需要查看历史工具调用记录时，请调用此工具。

        Args:
            tool_name(string): 工具名称。传空字符串或不传返回所有工具的调用概览；传具体工具名称返回该工具的详细调用记录（包含调用参数和返回结果）。
        '''
        session_id = event.unified_msg_origin
        records = self.compressed_records.get(session_id, [])

        if not records:
            return "当前会话没有被压缩的历史工具调用记录。若会话刚开始或尚未发生工具调用，这是正常的。"

        if not tool_name:
            summary: dict[str, int] = {}
            for r in records:
                summary[r["tool_name"]] = summary.get(r["tool_name"], 0) + 1
            lines = [
                f"当前会话共压缩了 {len(records)} 条历史工具调用记录，涉及以下工具：",
            ]
            for name, cnt in summary.items():
                lines.append(f"  - {name}（{cnt} 次）")
            lines.append("\n如需查看某工具的详细调用记录（包含参数和返回结果），请将工具名称作为 tool_name 参数再次调用本工具。")
            lines.append("⚠ 若你本意是查询某工具的详细记录却看到了此概览，请检查是否使用了正确的参数名 tool_name（传入 name、query 等其他参数名会被框架忽略）。")
            return "\n".join(lines)

        matched = [r for r in records if r["tool_name"] == tool_name]
        if not matched:
            available = sorted({r["tool_name"] for r in records})
            lines = [
                f"⚠ 未找到工具 '{tool_name}' 的压缩记录，可能是名称有误。",
                "当前有记录的工具名称：",
            ]
            for name in available:
                cnt = sum(1 for r in records if r["tool_name"] == name)
                lines.append(f"  - {name}（{cnt} 次）")
            lines.append("\n请使用上述工具名之一作为参数重新查询。")
            return "\n".join(lines)

        lines = [f"工具 '{tool_name}' 共有 {len(matched)} 条压缩记录："]
        for i, r in enumerate(matched, 1):
            lines.append(f"\n--- 记录 {i} ---")
            lines.append(f"触发请求：{r['user_query']}")
            lines.append(f"调用参数：{r['arguments']}")
            result_str = str(r["result"] or "（无结果）")
            if len(result_str) > 500:
                result_str = result_str[:500] + "…（已截断）"
            lines.append(f"返回结果：{result_str}")
        return "\n".join(lines)

    @filter.after_message_sent()
    async def on_session_cleared(self, event: AstrMessageEvent):
        """跟随系统 /reset 清空本插件的压缩记录缓存"""
        if event.get_extra("_clean_ltm_session", False):
            session_id = event.unified_msg_origin
            count = len(self.compressed_records.get(session_id, []))
            if count > 0:
                del self.compressed_records[session_id]
            self.round_counter[session_id] = 0
            logger.info(f"[压缩器] 检测到会话清空标记，已清理 {session_id} 的 {count} 条缓存记录")

    async def terminate(self):
        self.compressed_records.clear()
        logger.info("[压缩器] 插件已卸载，缓存已清理")
