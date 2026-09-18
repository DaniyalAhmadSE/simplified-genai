from simplified_genai import GptGenAiProvider


def test_gpt_temperature_support_blacklist() -> None:
    provider = GptGenAiProvider(api_key="test-key", default_model="gpt-4o")

    # Access private method for unit validation
    check_temp = getattr(provider, "_GptGenAiProvider__check_if_supports_temperature")

    # Blacklisted models (reasoning and non-nano GPT-5)
    assert check_temp("o1") is False
    assert check_temp("o1-mini") is False
    assert check_temp("o1-preview") is False
    assert check_temp("o3") is False
    assert check_temp("o3-mini") is False
    assert check_temp("o4-mini") is False
    assert check_temp("gpt-5.5") is False
    assert check_temp("gpt-5-pro") is False
    assert check_temp("gpt-5-reasoning") is False

    # Supported models
    assert check_temp("gpt-5-nano") is True
    assert check_temp("gpt-5.4-nano") is True
    assert check_temp("gpt-4o") is True
    assert check_temp("gpt-4o-mini") is True
    assert check_temp("gpt-4-turbo") is True
    assert check_temp("custom-model-id") is True
