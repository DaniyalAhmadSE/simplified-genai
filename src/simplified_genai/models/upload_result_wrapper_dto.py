from typing import Protocol

from simplified_genai.models.base_file_upload_result_dto import (
    BaseFileUploadResultDto,
)
from simplified_genai.models.gemini_upload_result_dto import (
    GeminiUploadResultDto,
)
from simplified_genai.models.gpt_upload_result_dto import GptUploadResultDto
from simplified_genai.models.grok_upload_result_dto import GrokUploadResultDto


class FileUploadResultWrapperDto[
    T: BaseFileUploadResultDto
    | GeminiUploadResultDto
    | GrokUploadResultDto
    | GptUploadResultDto
](Protocol):
    upload_result: T
