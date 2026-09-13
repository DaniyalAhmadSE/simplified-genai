from simplified_genai.models.base_file_upload_result_dto import (
    BaseFileUploadResultDto,
)


class GrokUploadResultDto(BaseFileUploadResultDto):
    uploaded_file_id: str
