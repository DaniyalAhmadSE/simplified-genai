from typing import override

from simplified_genai.interfaces.i_gen_ai_provider import IGenAiProvider
from simplified_genai.interfaces.i_gen_ai_provider_factory import (
    IGenAiProviderFactory,
)


class GenAiProviderFactory(IGenAiProviderFactory):
    def __init__(
        self,
        *,
        providers: list[IGenAiProvider],
        default_primary_gen_ai_provider_type: type[IGenAiProvider],
        default_fallback_gen_ai_provider_type: type[IGenAiProvider],
    ) -> None:
        self.__default_primary_gen_ai_provider_type = (
            default_primary_gen_ai_provider_type
        )
        self.__default_fallback_gen_ai_provider_type = (
            default_fallback_gen_ai_provider_type
        )

        self.__providers_by_type: dict[type[IGenAiProvider], IGenAiProvider] = {}

        for provider in providers:
            self.__providers_by_type[type(provider)] = provider

    @override
    def get_gen_ai_provider(
        self, provider_type: type[IGenAiProvider]
    ) -> IGenAiProvider:
        provider = self.__providers_by_type.get(provider_type)
        if not provider:
            raise ValueError(f"No provider registered with name: {provider_type}")
        return provider

    @override
    def set_primary_gen_ai_provider_type(
        self, gen_ai_provider_type: type[IGenAiProvider]
    ) -> None:
        self.__default_primary_gen_ai_provider_type = gen_ai_provider_type

    @override
    def set_fallback_gen_ai_provider_type(
        self, gen_ai_provider_type: type[IGenAiProvider]
    ) -> None:
        self.__default_fallback_gen_ai_provider_type = gen_ai_provider_type

    @override
    def get_primary_gen_ai_provider(self) -> IGenAiProvider:
        return self.get_gen_ai_provider(self.__default_primary_gen_ai_provider_type)

    @override
    def get_fallback_gen_ai_provider(self) -> IGenAiProvider:
        return self.get_gen_ai_provider(self.__default_fallback_gen_ai_provider_type)
