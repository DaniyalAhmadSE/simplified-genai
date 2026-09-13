from typing import Any

from simplified_genai.models.base_file_upload_result_dto import (
    BaseFileUploadResultDto,
)


class GptUploadResultDto(BaseFileUploadResultDto):
    file_payload: dict[str, Any]
