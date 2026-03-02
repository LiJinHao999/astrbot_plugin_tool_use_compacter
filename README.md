# 工具调用压缩器 (astrbot_plugin_tool_use_compacter)

基于 [@zz6zz666](https://github.com/zz6zz666) 的 [astrbot_plugin_tool_use_cleaner](https://github.com/zz6zz666/astrbot_plugin_tool_use_cleaner) 改进而来。

一个 AstrBot 插件，将历史工具调用记录从 LLM 请求上下文中压缩移除以减少 token 浪费，同时缓存这些记录并注册一个 LLM Tool，让大模型在需要时可以主动回溯查询被压缩的工具调用详情。

## 工作原理

1. **压缩**：在每次 LLM 请求前（`on_llm_request`），从上下文中提取工具调用记录（assistant 的 `tool_calls` + tool 的响应），缓存后移除
2. **回溯**：注册 `query_compressed_tools` 工具，LLM 可主动调用查询被压缩的记录概览或某个工具的详细调用参数和返回结果
3. **重置检测**：监听所有消息，匹配重置关键词时自动清空该会话的压缩缓存（不拦截消息传播）

会话隔离基于 `unified_msg_origin`，不同群组/私聊互不干扰。

## 配置选项

| 配置项 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| tool_context_keep_rounds | int | 0 | 上下文中保留最近 N 轮的工具调用信息（0=全部清除） |
| compressed_record_keep_rounds | int | 0 | 压缩缓存保留最近 N 轮的记录供 LLM 查询（0=不限制） |
| session_reset_keywords | list | ["reset","/reset","clear","/clear","重置会话", "清空上下文", "新对话"] | 触发清空压缩缓存的关键词列表 |

## 安装

- 在 AstrBot 插件市场搜索安装
- 通过 WebUI 从链接或文件安装

## 版本历史

### v2.0.0 
- 从"清洗"升级为"压缩"：移除的工具调用记录缓存到内存，支持 LLM 主动回溯查询
- 新增 `query_compressed_tools` LLM Tool
- 新增压缩缓存轮数控制（`compressed_record_keep_rounds`）
- 新增会话重置关键词检测（`session_reset_keywords`）
- 基于请求轮次计数器的精确缓存淘汰

### v1.3.0
- 支持按轮数控制工具调用信息的保留范围

### v1.1.0
- assistant 消息同时包含 content 和 tool_calls 时，保留 content 仅剔除 tool_calls

### v1.0.0
- 初始版本，支持过滤工具调用请求和响应

## 作者

[@Lijinhao999](https://github.com/Lijinhao999)

原始项目：[@zz6zz666/astrbot_plugin_tool_use_cleaner](https://github.com/zz6zz666/astrbot_plugin_tool_use_cleaner)

## 许可证

GNU Affero General Public License v3.0
