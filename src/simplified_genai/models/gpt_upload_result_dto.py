from typing import Any

from simplified_genai.models.file_upload_response import (
    FileUploadResponse,
)


class GptUploadResultDto(FileUploadResponse):
    file_payload: dict[str, Any]
