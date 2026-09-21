from langchain_core.callbacks import UsageMetadataCallbackHandler


def total_input_tokens(usage: UsageMetadataCallbackHandler) -> int:
    return sum(item["input_tokens"] for item in usage.usage_metadata.values())


def total_output_tokens(usage: UsageMetadataCallbackHandler) -> int:
    return sum(item["output_tokens"] for item in usage.usage_metadata.values())
