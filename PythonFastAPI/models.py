from database import Base
from sqlalchemy import  Column, Integer, String, Text, TIMESTAMP, ForeignKey, Boolean, JSON
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String, nullable=False, unique=True)
    email = Column(String, nullable=False, unique=True)
    hashed_pwd = Column(String, nullable=False)
    created_at = Column(TIMESTAMP, nullable=False)

    session_logs = relationship("SessionLogs", back_populates="user")
    data_storage = relationship("DataStorage", back_populates="user")
    conversations = relationship("LLMConversation", back_populates="")
    illustrations = relationship("ImageGeneration", back_populates="user")
    detections = relationship("ImageDetection", back_populates="user")
    classifications = relationship("ImageClassification", back_populates="user")
    vocalizations = relationship("VoiceGeneration", back_populates="user")
    recognitions = relationship("VoiceRecognition", back_populates="user")

class Inquiries(Base):
    __tablename__ = 'inquiries'

    id = Column(Integer, primary_key=True, autoincrement=True)
    resolved = Column(Boolean, nullable=False, default=False)
    contact_name = Column(String, nullable=False)
    contact_email = Column(String, nullable=False)
    contact_message = Column(Text, nullable=False)

class SessionLogs(Base):
    __tablename__ = 'sessions'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'))
    jwt = Column(String, nullable=False)
    start_time = Column(TIMESTAMP, nullable=False)

    user = relationship("User", back_populates="session_logs")


class DataStorage(Base):
    __tablename__ = 'data_storage'
    
    data_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'))
    created_at = Column(TIMESTAMP, default=datetime.utcnow())

    text_content = Column(String, nullable=True)
    audio_path = Column(String, nullable=True)
    image_path = Column(String, nullable=True)
    audio_path = Column(String, nullable=True)
    document_path = Column(String, nullable=True)
    meta = Column(JSON, nullable=True) 

    user = relationship("User", back_populates="data_storage")
 
    

class LLMConversation(Base):
    __tablename__ = 'conversations'

    convo_uuid = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    has_context = Column(Boolean, default=False)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'))
    created_at = Column(TIMESTAMP, default=datetime.utcnow())
    updated_at = Column(TIMESTAMP, default=datetime.utcnow(), onupdate=datetime.utcnow())
    num_messages = Column(Integer, default=0)

    user = relationship("User", back_populates="conversations")

class LLMConversationMessage(Base):
    __tablename__ = 'conversation_messages'

    id = Column(Integer, primary_key=True)
    convo_id = Column(String, ForeignKey('conversations.convo_uuid', ondelete='CASCADE'))
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'))
    created_at = Column(TIMESTAMP, default=datetime.utcnow())
  
    audio_file_path = Column(String, nullable=True)
    document_file_path = Column(String, nullable=True)

    message_prompt = Column(Text, nullable=True)
    message_response = Column(Text, nullable=True)


#

class ImageGeneration(Base):
    __tablename__ = 'illustrations'

    illustration_uuid = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'))
    created_at = Column(TIMESTAMP, default=datetime.utcnow())

    prompt = Column(String, nullable=False)
    illustration_path = Column(String, nullable=False) 

    user = relationship("User", back_populates="illustrations")

#

class ImageDetection(Base):
    __tablename__ = 'detections'

    detection_uuid = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'))
    created_at = Column(TIMESTAMP, default=datetime.utcnow())

    source_path = Column(String, nullable=False)
    detection = Column(String, nullable=False) 
    detection_path = Column(String, nullable=False)

    user = relationship("User", back_populates="detections")

#
class ImageClassification(Base):
    __tablename__ = 'classifications'

    classification_uuid = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'))
    created_at = Column(TIMESTAMP, default=datetime.utcnow())

    source_path = Column(String, nullable=False)
    categories = Column(String, nullable=False)
    classification = Column(String, nullable=False) 

    user = relationship("User", back_populates="classifications")

#

class VoiceGeneration(Base):
    __tablename__ = 'speech'

    speech_uuid = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'))
    created_at = Column(TIMESTAMP, default=datetime.utcnow())

    audio_path = Column(String, nullable=False)
    speech_prompt = Column(String, nullable=False) 
    speech_path = Column(String, nullable=False)

    user = relationship("User", back_populates="vocalizations")

#

class VoiceRecognition(Base):
    __tablename__ = 'recognitions'

    recognition_uuid = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'))
    created_at = Column(TIMESTAMP, default=datetime.utcnow())

    audio_path = Column(String, nullable=False)
    recognition = Column(String, nullable=False) 

    user = relationship("User", back_populates="recognitions")