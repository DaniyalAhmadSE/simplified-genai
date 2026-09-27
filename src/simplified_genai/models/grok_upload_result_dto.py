from simplified_genai.models.file_upload_response import (
    FileUploadResponse,
)


class GrokUploadResultDto(FileUploadResponse):
    uploaded_file_id: str
