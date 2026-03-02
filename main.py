from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api.provider import ProviderRequest
from astrbot.api import logger, AstrBotConfig
from collections import defaultdict


@register("tool_use_compacter", "Lijinhao999", "参考自astrbot_plugin_tool_use_cleaner，清除历史工具调用记录以减少token浪费，并提供LLM可查询的回溯能力。支持缓存调用记录轮数控制、重置关键词检测。", "2.0.0")
class ToolUseCleanerPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.tool_context_keep_rounds = int(self.config.get("tool_context_keep_rounds", 0))
        self.compressed_keep_rounds = int(self.config.get("compressed_record_keep_rounds", 0))
        self.reset_keywords: list[str] = self.config.get("session_reset_keywords", ["reset", "/reset", "clear", "/clear", "重置会话", "清空上下文", "新对话"])
        # 会话隔离的压缩记录缓存: unified_msg_origin -> list[record]
        self.compressed_records: dict[str, list[dict]] = defaultdict(list)
        # 每个会话的请求轮次计数器
        self.round_counter: dict[str, int] = defaultdict(int)
        keep_str = "全部清除" if self.tool_context_keep_rounds <= 0 else f"保留最近{self.tool_context_keep_rounds}轮"
        cache_str = "不限轮数" if self.compressed_keep_rounds <= 0 else f"最近{self.compressed_keep_rounds}轮"
        logger.info(f"[压缩器] 插件已初始化：上下文工具调用{keep_str}，压缩缓存保留{cache_str}")

    def _find_round_ends(self, contexts: list[dict]) -> list[int]:
        """找到所有轮次的结束位置索引（assistant->user/system 转换点）"""
        round_ends = []
        for i in range(len(contexts) - 1):
            if contexts[i].get("role") == "assistant" and contexts[i + 1].get("role") in ["user", "system"]:
                round_ends.append(i)
        if contexts and contexts[-1].get("role") == "assistant":
            round_ends.append(len(contexts) - 1)
        return round_ends

    def _extract_tool_records(self, contexts: list[dict], start: int, end: int, round_num: int) -> list[dict]:
        """从指定范围的上下文中提取工具调用记录（调用+响应配对+触发消息）"""
        tool_calls_map = {}
        last_user_msg = ""
        for i in range(start, min(end + 1, len(contexts))):
            ctx = contexts[i]
            if ctx.get("role") == "user":
                content = ctx.get("content", "")
                last_user_msg = content if isinstance(content, str) else str(content)
            elif ctx.get("role") == "assistant" and "tool_calls" in ctx:
                for tc in ctx["tool_calls"]:
                    call_id = tc.get("id", "")
                    func = tc.get("function", {})
                    if func.get("name") == "query_compressed_tools":
                        continue
                    tool_calls_map[call_id] = {
                        "tool_name": func.get("name", "unknown"),
                        "tool_call_id": call_id,
                        "arguments": func.get("arguments", ""),
                        "result": None,
                        "user_query": last_user_msg,
                        "round_num": round_num,
                    }
            elif ctx.get("role") == "tool":
                call_id = ctx.get("tool_call_id", "")
                if call_id in tool_calls_map:
                    tool_calls_map[call_id]["result"] = ctx.get("content", "")
        return list(tool_calls_map.values())

    def _store_records(self, session_id: str, records: list[dict]):
        """存储压缩记录，按 tool_call_id 去重"""
        if not records:
            return
        existing_ids = {r["tool_call_id"] for r in self.compressed_records[session_id]}
        new_records = [r for r in records if r["tool_call_id"] not in existing_ids]
        if not new_records:
            return
        self.compressed_records[session_id].extend(new_records)
        total = len(self.compressed_records[session_id])
        logger.info(f"[压缩器] 会话 ...{session_id[-12:]} 新增 {len(new_records)} 条压缩记录，累计 {total} 条")

    def _trim_records_by_rounds(self, session_id: str):
        """按轮数限制淘汰过旧的压缩记录"""
        if self.compressed_keep_rounds <= 0 or not self.compressed_records[session_id]:
            return
        cutoff = self.round_counter[session_id] - self.compressed_keep_rounds
        before = len(self.compressed_records[session_id])
        self.compressed_records[session_id] = [
            r for r in self.compressed_records[session_id] if r["round_num"] > cutoff
        ]
        trimmed = before - len(self.compressed_records[session_id])
        if trimmed > 0:
            logger.debug(f"[压缩器] 会话 ...{session_id[-12:]} 淘汰 {trimmed} 条旧记录，剩余 {len(self.compressed_records[session_id])} 条")

    @filter.on_llm_request()
    async def clean_context(self, event: AstrMessageEvent, req: ProviderRequest):
        """压缩上下文中的工具调用：提取记录到缓存后再从请求体中移除"""
        if not req.contexts:
            return

        session_id = event.unified_msg_origin
        original_count = len(req.contexts)
        round_ends = self._find_round_ends(req.contexts)
        self.round_counter[session_id] += 1
        current_round = self.round_counter[session_id]

        # 收集自身工具的 tool_call_id，清理时保留这些消息
        self_tool_ids = set()
        for ctx in req.contexts:
            if ctx.get("role") == "assistant" and "tool_calls" in ctx:
                for tc in ctx["tool_calls"]:
                    if tc.get("function", {}).get("name") == "query_compressed_tools":
                        self_tool_ids.add(tc.get("id", ""))

        if self.tool_context_keep_rounds <= 0:
            # 全部清除模式：提取全量工具调用后清除
            records = self._extract_tool_records(req.contexts, 0, len(req.contexts) - 1, current_round)
            self._store_records(session_id, records)

            cleaned_contexts = []
            for ctx in req.contexts:
                if ctx.get("role") == "tool":
                    if ctx.get("tool_call_id") in self_tool_ids:
                        cleaned_contexts.append(ctx)
                    continue
                if ctx.get("role") == "assistant":
                    if "tool_calls" in ctx:
                        self_calls = [tc for tc in ctx["tool_calls"] if tc.get("id") in self_tool_ids]
                        other_calls = [tc for tc in ctx["tool_calls"] if tc.get("id") not in self_tool_ids]
                        if other_calls and not ctx.get("content") and not self_calls:
                            continue
                        cleaned_ctx = ctx.copy()
                        if self_calls:
                            cleaned_ctx["tool_calls"] = self_calls
                        else:
                            del cleaned_ctx["tool_calls"]
                        cleaned_contexts.append(cleaned_ctx)
                        continue
                    elif not ctx.get("content"):
                        continue
                cleaned_contexts.append(ctx)
            req.contexts = cleaned_contexts
        else:
            # 保留N轮模式：只清理超出范围的旧轮次
            # 修复：需保留最近 K 轮，当历史轮数 <= K 时无需清理
            # 当历史轮数 > K 时，清理到第 (N-K-1) 个 round_end（即保留后K轮）
            cutoff_index = -1
            if round_ends:
                if len(round_ends) <= self.tool_context_keep_rounds:
                    # 历史轮数不超过保留限制，无需清理
                    cutoff_index = -1
                    logger.debug(f"[压缩器] 会话 ...{session_id[-12:]} 第{current_round}轮：共{len(round_ends)}轮 ≤ 保留{self.tool_context_keep_rounds}轮，跳过清理")
                else:
                    # 清理最旧的 (N-K) 轮，保留最新 K 轮
                    cutoff_index = round_ends[-(self.tool_context_keep_rounds + 1)]
                    logger.debug(f"[压缩器] 会话 ...{session_id[-12:]} 第{current_round}轮：共{len(round_ends)}轮，保留{self.tool_context_keep_rounds}轮，清理到索引{cutoff_index}")

            if cutoff_index >= 0:
                records = self._extract_tool_records(req.contexts, 0, cutoff_index, current_round)
                self._store_records(session_id, records)

            new_contexts = []
            for i, ctx in enumerate(req.contexts):
                if i <= cutoff_index:
                    if ctx.get("role") == "tool":
                        if ctx.get("tool_call_id") in self_tool_ids:
                            new_contexts.append(ctx)
                        continue
                    if ctx.get("role") == "assistant" and "tool_calls" in ctx:
                        self_calls = [tc for tc in ctx["tool_calls"] if tc.get("id") in self_tool_ids]
                        cleaned_ctx = ctx.copy()
                        if self_calls:
                            cleaned_ctx["tool_calls"] = self_calls
                        else:
                            del cleaned_ctx["tool_calls"]
                        new_contexts.append(cleaned_ctx)
                        continue
                new_contexts.append(ctx)
            req.contexts = new_contexts

        # 按轮数限制淘汰旧的压缩记录
        self._trim_records_by_rounds(session_id)

        removed_count = original_count - len(req.contexts)
        cached_total = len(self.compressed_records.get(session_id, []))
        if removed_count > 0:
            logger.info(f"[压缩器] 会话 ...{session_id[-12:]} 第{current_round}轮：上下文 {original_count}→{len(req.contexts)} 条（移除{removed_count}条），缓存共{cached_total}条")
        else:
            logger.debug(f"[压缩器] 会话 ...{session_id[-12:]} 第{current_round}轮：无工具调用需压缩（{original_count}条），缓存共{cached_total}条")

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def detect_reset(self, event: AstrMessageEvent):
        """检测会话重置关键词，清空该会话的压缩记录缓存（不拦截消息）"""
        msg = event.message_str.strip()
        session_id = event.unified_msg_origin
        for keyword in self.reset_keywords:
            if keyword and msg == keyword:
                if session_id in self.compressed_records:
                    count = len(self.compressed_records[session_id])
                    del self.compressed_records[session_id]
                    self.round_counter[session_id] = 0
                    logger.info(f"[压缩器] 检测到重置关键词 '{keyword}'，已清空会话 ...{session_id[-12:]} 的 {count} 条压缩记录")
                break
        # 不调用 event.stop_event()，让消息继续传播

    @filter.llm_tool(name="query_compressed_tools")
    async def query_compressed_tools(self, event: AstrMessageEvent, tool_name: str = ""):
        '''你在会话中的历史调用工具记录将会被压缩删除。此工具用于查询当前会话中被压缩的历史工具调用记录，包括触发工具的消息、调用参数和返回结果。当你需要查看会话中的历史工具调用记录时，请调用此工具。

            tool_name(string): 工具名称，传空字符串""返回所有工具的调用概览，传具体工具名称返回该工具的详细调用记录
        '''
        session_id = event.unified_msg_origin
        records = self.compressed_records.get(session_id, [])

        if not records:
            return "当前会话没有被压缩的历史工具调用记录。（若会话刚开始或未发生过工具调用，这是正常现象）"

        if not tool_name:
            # 空传或参数不匹配时均返回概览（防呆）
            summary: dict[str, int] = {}
            for r in records:
                summary[r["tool_name"]] = summary.get(r["tool_name"], 0) + 1
            lines = [f"当前会话共有 {len(records)} 条被压缩的历史工具调用记录："]
            for name, cnt in summary.items():
                lines.append(f"- {name}: {cnt} 次")
            lines.append("\n如需查看某工具的详细调用参数和返回结果，请将该工具名称作为 tool_name 参数再次调用本工具。")
            return "\n".join(lines)

        matched = [r for r in records if r["tool_name"] == tool_name]
        if not matched:
            # 防呆：名称不存在时，返回警告 + 概览
            available = sorted({r["tool_name"] for r in records})
            summary_lines = [f"⚠ 未找到工具 '{tool_name}' 的压缩记录，以下是当前有记录的工具："]
            for name in available:
                cnt = sum(1 for r in records if r["tool_name"] == name)
                summary_lines.append(f"- {name}: {cnt} 次")
            summary_lines.append("\n请使用上述工具名之一作为 tool_name 参数重新查询。")
            return "\n".join(summary_lines)

        lines = [f"工具 '{tool_name}' 共有 {len(matched)} 条压缩记录："]
        for i, r in enumerate(matched, 1):
            lines.append(f"\n--- 记录 {i} ---")
            lines.append(f"触发请求: {r['user_query']}")
            lines.append(f"调用参数: {r['arguments']}")
            result_str = str(r["result"] or "（无结果）")
            if len(result_str) > 500:
                result_str = result_str[:500] + "...(已截断)"
            lines.append(f"返回结果: {result_str}")
        return "\n".join(lines)

    async def terminate(self):
        """插件卸载时清理缓存"""
        self.compressed_records.clear()
        logger.info("[压缩器] 插件已卸载，缓存已清理")
