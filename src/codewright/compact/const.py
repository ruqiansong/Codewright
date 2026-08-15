"""Internal constants for context management."""

# 单条工具结果触发落盘的字节阈值。
SINGLE_RESULT_LIMIT = 50_000

# 单条工具结果消息允许保留的聚合字节上限。
MESSAGE_AGGREGATE_LIMIT = 200_000

# 为摘要模型输出预留的 token 数。
SUMMARY_RESERVE = 20_000

# 自动压缩用于吸收估算误差的 token 余量。
AUTO_SAFETY_MARGIN = 13_000

# 手动压缩请求的 token 安全余量。
MANUAL_SAFETY_MARGIN = 3_000

# 恢复段最多包含的近期文件数量。
RECOVERY_FILE_LIMIT = 5

# 恢复段中单个文件允许占用的估算 token 数。
RECOVERY_TOKENS_PER_FILE = 5_000

# 摘要后近期原文需要保留的估算 token 下界。
RECENT_KEEP_TOKENS = 10_000

# 摘要后近期原文需要保留的消息条数下界。
RECENT_KEEP_MESSAGES = 5

# 自动摘要连续失败后触发熔断的次数。
MAX_CONSECUTIVE_AUTO_COMPACT_FAILURES = 3

# 摘要请求超长时逐组直接重试的次数。
PTL_RETRY_LIMIT = 3

# 逐组重试耗尽后每次丢弃的剩余消息组比例。
PTL_DROP_PERCENTAGE = 0.2

# 无精确 tokenizer 时使用的平均字节 token 比。
ESTIMATE_CHARS_PER_TOKEN = 3.5

# 工具结果预览允许保留的头部字节上限。
PREVIEW_HEAD_BYTES = 2_048

# 工具结果预览允许保留的头部行数上限。
PREVIEW_HEAD_LINES = 20
