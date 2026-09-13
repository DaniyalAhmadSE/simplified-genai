from google.genai import types

from simplified_genai.models.base_file_upload_result_dto import (
    BaseFileUploadResultDto,
)


class GeminiUploadResultDto(BaseFileUploadResultDto):
    uploaded_file: types.File
