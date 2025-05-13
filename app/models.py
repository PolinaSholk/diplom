import hashlib
from datetime import datetime

from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime, LargeBinary

from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)

class UploadedFile(Base):
    __tablename__ = "uploaded_files"


    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    filepath = Column(String)  # Путь к сохраненному файлу
    owner_id = Column(Integer, ForeignKey("users.id"))
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    checksum = Column(String)  # MD5 хеш файла
    file_content = Column(LargeBinary)


    @classmethod
    def calculate_checksum(cls, file_path):
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()