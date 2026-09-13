from pydantic import BaseModel, ConfigDict


class FileDto(BaseModel):
    file_binary: bytes
    mime_type: str

    model_config = ConfigDict(arbitrary_types_allowed=True)
