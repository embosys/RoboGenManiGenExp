# AGENTS.md

## CodeGraph

本仓库存在 `.codegraph/` 时，理解或定位代码优先使用 CodeGraph：
- 有 MCP 工具时，使用 `codegraph_explore`。
- 否则，在仓库根目录执行 `codegraph explore "<符号名或问题>"`。
- 索引过期时，执行 `codegraph sync`。
- 需要完全重建索引时，执行 `codegraph index`。

`.codegraph/` 是本地生成的索引，不要提交到 Git。
