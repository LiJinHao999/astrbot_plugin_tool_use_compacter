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
        logger.info(f"工具调用压缩插件已初始化，上下文保留{self.tool_context_keep_rounds}轮，压缩缓存保留{self.compressed_keep_rounds}轮")

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
        """存储压缩记录，按 tool_call_id 去重，按轮数限制淘汰"""
        if not records:
            return
        existing_ids = {r["tool_call_id"] for r in self.compressed_records[session_id]}
        new_records = [r for r in records if r["tool_call_id"] not in existing_ids]
        if not new_records:
            return
        self.compressed_records[session_id].extend(new_records)
        logger.debug(f"会话 {session_id} 新增 {len(new_records)} 条压缩记录")

    def _trim_records_by_rounds(self, session_id: str):
        """按轮数限制淘汰过旧的压缩记录"""
        if self.compressed_keep_rounds <= 0 or not self.compressed_records[session_id]:
            return
        cutoff = self.round_counter[session_id] - self.compressed_keep_rounds
        self.compressed_records[session_id] = [
            r for r in self.compressed_records[session_id] if r["round_num"] > cutoff
        ]

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
            # 提取全部工具调用记录后清除
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
                        # 保留自身工具的 tool_calls，移除其他的
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
            # 按轮次控制清理范围
            cutoff_index = -1
            if round_ends:
                if len(round_ends) < self.tool_context_keep_rounds:
                    cutoff_index = round_ends[0]
                else:
                    cutoff_index = round_ends[-self.tool_context_keep_rounds]

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
        if removed_count > 0:
            logger.info(f"上下文压缩: 移除了 {removed_count} 条消息，记录已缓存")

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
                    logger.info(f"检测到重置关键词 '{keyword}'，已清空会话 {session_id} 的 {count} 条压缩记录")
                break
        # 不调用 event.stop_event()，让消息继续传播

    @filter.llm_tool(name="query_compressed_tools")
    async def query_compressed_tools(self, event: AstrMessageEvent, tool_name: str = ""):
        '''你在会话中的历史调用工具记录将会被压缩删除。此工具用于查询当前会话中被压缩的历史工具调用记录，包括触发工具的消息、调用参数和返回结果。当你需要查看会话中的历史工具调用记录时，请调用此工具。    
        
            tool_name(string): 工具名称，传空字符串""返回概览，传具体名称返回详细记录
        '''
        session_id = event.unified_msg_origin
        records = self.compressed_records.get(session_id, [])

        if not records:
            yield event.plain_result("当前会话没有被压缩的工具调用记录。")
            return

        if not tool_name:
            summary = {}
            for r in records:
                name = r["tool_name"]
                summary[name] = summary.get(name, 0) + 1
            lines = [f"当前会话共有 {len(records)} 条被压缩的历史工具调用记录："]
            for name, count in summary.items():
                lines.append(f"- {name}: {count} 次调用")
            lines.append("\n可通过传入具体工具名查询详细的调用参数和返回结果。")
            yield event.plain_result("\n".join(lines))
        else:
            matched = [r for r in records if r["tool_name"] == tool_name]
            if not matched:
                yield event.plain_result(f"未找到工具 '{tool_name}' 的压缩记录。")
                return
            lines = [f"工具 '{tool_name}' 共有 {len(matched)} 条压缩记录："]
            for i, r in enumerate(matched, 1):
                lines.append(f"\n--- 记录 {i} ---")
                lines.append(f"用户发送过请求: {r['user_query']}")
                lines.append(f"你在那时候调用了工具: {r['arguments']}")
                result_str = str(r["result"] or "")
                if len(result_str) > 500:
                    result_str = result_str[:500] + "...(截断)"
                lines.append(f"返回结果: {result_str}")
            yield event.plain_result("\n".join(lines))

    async def terminate(self):
        """插件卸载时清理缓存"""
        self.compressed_records.clear()
        logger.info("工具调用压缩插件已卸载")