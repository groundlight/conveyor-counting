from enum import Enum

class StrEnum(str, Enum):
    @classmethod
    def get_values(cls) -> list[str]:
        return [value.value for value in cls]
    
    @classmethod
    def get_default(cls) -> str:
        return cls.get_values()[0]

class RecordingMode(StrEnum):
    NONE = "NONE"
    RAW = "RAW"
    ANNOTATED = "ANNOTATED"

class AppMode(StrEnum):
    VIDEO_ONLY = "VIDEO_ONLY"
    VIDEO_INFERENCE = "VIDEO_INFERENCE"
    SNAPSHOT_INFERENCE = "SNAPSHOT_INFERENCE"