from google.genai import types

from simplified_genai.models.file_upload_response import (
    FileUploadResponse,
)


class GeminiUploadResultDto(FileUploadResponse):
    uploaded_file: types.File
