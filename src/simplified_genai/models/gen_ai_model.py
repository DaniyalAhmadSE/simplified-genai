from enum import StrEnum


class GenAiModel(StrEnum):
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
    GEMINI_3_1_FLASH_LITE = "gemini-3.1-flash-lite"
    GEMINI_3_5_FLASH = "gemini-3.5-flash"
    GEMINI_3_6_FLASH = "gemini-3.6-flash"
    GEMINI_3_7_FLASH = "gemini-3.7-flash"
    GROK_4_1_FAST_REASONING = "grok-4-1-fast-reasoning"
    GPT_5_NANO = "gpt-5-nano"
    GPT_5_4_NANO = "gpt-5.4-nano"
    GPT_5_5 = "gpt-5.5"
