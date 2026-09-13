from typing import Protocol

from simplified_genai.interfaces.i_gen_ai_provider import IGenAiProvider


class IGenAiProviderFactory(Protocol):
    def get_gen_ai_provider(
        self, provider_type: type[IGenAiProvider]
    ) -> IGenAiProvider: ...

    def set_primary_gen_ai_provider_type(
        self, gen_ai_provider_type: type[IGenAiProvider]
    ) -> None: ...

    def set_fallback_gen_ai_provider_type(
        self, gen_ai_provider_type: type[IGenAiProvider]
    ) -> None: ...

    def get_primary_gen_ai_provider(self) -> IGenAiProvider: ...

    def get_fallback_gen_ai_provider(self) -> IGenAiProvider: ...
