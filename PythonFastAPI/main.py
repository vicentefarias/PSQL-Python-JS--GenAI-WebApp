# Standard library imports
import os
import json
import csv
from io import StringIO
import re
import uuid
from datetime import datetime, timedelta

# Third-party imports
import jwt
from jose import JWTError
import aiofiles
from email_validator import validate_email, EmailNotValidError
from passlib.context import CryptContext

# FastAPI imports
from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer

# SQLAlchemy imports
from sqlalchemy.orm import Session

# Pydantic imports
from pydantic import BaseModel

# Typing imports
from typing import Annotated, List, Optional

# Local application imports
from database import SessionLocal, engine
import models as models

# Inference imports
from inference import (
    get_LLM_response,
    get_document_response,
    split_documents,
    add_to_chroma,
    get_RAG_response,
    get_diffusion_response,
    get_detection_response,
    get_classification_response,
    get_speech_response,
    get_recognition_response,
)


# FastAPI Server Setup

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()

origins = ['http://localhost:3000']

app.add_middleware(CORSMiddleware, 
                   allow_origins=origins,  
                   allow_credentials=True, 
                   allow_methods=['*'],
                   allow_headers=['*'])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]

models.Base.metadata.create_all(bind=engine)

#
#   Functions for Endpoint 1: POST /register/
#

class RegisterBase(BaseModel):
    uname: str
    email: str
    pwd: str
    pwd_conf: str

class RegisterModel(RegisterBase):
    user_id: int
    class Config:
        orm_mode = True

SECRET_KEY = "<SECRET_KEY>"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def valid_email(email: str):
    try:
        v = validate_email(email)
        return True
    except EmailNotValidError as e:
        return False

def valid_password(password: str):
    if len(password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters long."
        )
    if not re.search(r"[A-Z]", password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must contain at least one uppercase letter."
        )
    if not re.search(r"[a-z]", password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must contain at least one lowercase letter."
        )
    if not re.search(r"\d", password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must contain at least one digit."
        )
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must contain at least one special character."
        )
    if re.search(r"\s", password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must not contain any spaces."
        )
    return True

def valid_username(username: str):
    # Check that the username is between 3 and 20 characters long
    if len(username) < 3 or len(username) > 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username must be between 3 and 20 characters long."
        )
    
    # Check that the username only contains alphanumeric characters and underscores
    if not re.match(r"^[a-zA-Z0-9_]+$", username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username can only contain letters, numbers, and underscores."
        )
    
    return True

def user_or_email_existance(db: Session, username: str, email: str):
    user_by_username = db.query(models.User).filter(models.User.username == username).all()
    user_by_email = db.query(models.User).filter(models.User.email == email).all()

    if user_by_username or user_by_email:
        return True 
        
    return False 

def string_hash(string: str):
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    return pwd_context.hash(string)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire.timestamp()})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_session_log(user_id: int, hashed_token: str, db: db_dependency):
    id = db.query(models.SessionLogs).count()+1
    db_session = models.SessionLogs(id=id, user_id=user_id, jwt=hashed_token, start_time=datetime.utcnow())
    db.add(db_session)
    db.commit()
    db.refresh(db_session)

#
#   Functions for Endpoint 2: POST /login/
#

class LoginBase(BaseModel):
    email: str
    pwd: str

class LoginModel(LoginBase):
    id: int
    class Config:
        orm_mode = True


def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def verify_password(unhashed_pwd, hashed_pwd):
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    return pwd_context.verify(unhashed_pwd, hashed_pwd)

class TokenData(BaseModel):
    username: str | None = None

def verify_jwt(token: str):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except Exception as e:
        raise credentials_exception

#
#   Functions for Endpoint 3: POST /contact/
#

class InquiryCreate(BaseModel):
    contact_name: str
    contact_email: str
    contact_message: str


#
#   Functions for Endpoint 4: POST /user/<user_id>/dataStore/
#

async def build_data_store_path(user: str, data_id: str, file_type: str, file: UploadFile) -> str:
    extension = os.path.splitext(file.filename)[1]  # Get the file extension
    return f'./users/{user}/dataStore/{data_id}/{file_type}{extension}'

#
#   Functions for Endpoint 5: POST /users/<user_id>/conversation/
#

def get_user_id(username: str, db: Session):
    user = db.query(models.User).filter(models.User.username == username).one()
    return user.id

def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def build_conversation_file_path(user: str, convo_uuid: str, message_id: int, file_type: str, file: UploadFile):
        return f'./users/{user}/conversations/{convo_uuid}/message/{message_id}/{file_type}/{file.filename}'

async def save_file_to_path(file: UploadFile, path: str):
    try:
        with open(path, 'wb') as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    

#
#   Functions for Endpoint 6: POST /users/<user_id>/conversation/convo_uuid/message/
#



def get_conversation_from_id(convo_uuid: str, db: Session) -> models.LLMConversation:
    try:
        conversation = db.query(models.LLMConversation).filter_by(convo_uuid=convo_uuid).one_or_none()
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating conversation message: {str(e)}")
    
def format_llm_message_history(convo_uuid: str, db: Session) -> str:
    # Query the conversation messages
    messages = db.query(models.LLMConversationMessage).filter_by(convo_id=convo_uuid).order_by(models.LLMConversationMessage.id).all()

    if not messages:
        return ""

    # Format messages
    formatted_history = ""
    for msg in messages:
        formatted_history += f"User: {msg.message_prompt}\nLLM: {msg.message_response}\n\n"

    return formatted_history 

#---------------------------------------------------------------------------------------------------------------------------------------------------------

#
#   Endpoint 1: POST /register
#

@app.post("/register/")
async def register_user(register: RegisterBase, db: db_dependency):
    uname, email, pwd, pwd_conf = register.uname,  register.email, register.pwd, register.pwd_conf
    if not valid_email(email=email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email must be valid."
        )
  
    if not valid_password(password=pwd):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must adhere to constraints."
        )
    
    valid_username(uname)

    if pwd != pwd_conf:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be identical."
        )

    if user_or_email_existance(db=db, email=email, username=uname):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username and/or email already exists."
        )

    hashed_pwd = string_hash(pwd)
    user_id = db.query(models.User).count()+1

    db_user = models.User(id=user_id, username=uname, email=email, hashed_pwd=hashed_pwd, created=datetime.now())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    access_token = create_access_token(data={"sub": db_user.username})
    create_session_log(user_id, string_hash(access_token), db)

    return {"user": db_user, "access_token": access_token, "token_type": "bearer"}

#
#   Endpoint 2: POST /login
#

@app.post("/login/")
async def login_user(login: LoginBase, db:  Session = Depends(get_db)):
    user = get_user_by_email(db, login.email)
    if not user or not verify_password(login.pwd, user.hashed_pwd):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    create_session_log(user.id, string_hash(access_token), db)
    return {"user": user, "access_token": access_token, "token_type": "bearer"}
    

#   
#   Endpoint 3: POST /contact
#

@app.post("/contact")
async def create_inquiry(inquiry: InquiryCreate, db: Session = Depends(get_db)):
    # Create a new Inquiry instance
    db_inquiry = models.Inquiries(
        contact_name=inquiry.contact_name,
        contact_email=inquiry.contact_email,
        contact_message=inquiry.contact_message
    )

    try:
        valid_email(inquiry.contact_email)
        # Add the new inquiry to the session
        db.add(db_inquiry)
        # Commit the transaction
        await db.commit()
        # Refresh the instance to get the auto-generated ID
        await db.refresh(db_inquiry)
    except Exception as e:
        # Rollback the transaction in case of an error
        await db.rollback()
        # Raise an HTTPException with a 500 status code
        raise HTTPException(status_code=500, detail=str(e))

    # Return the created inquiry
    return "Successfully submitted inquiry."

#   Data Storage endpoint for authorized users
#   Endpoint 4: POST /users/{user}/dataStore
#


@app.post("/users/{user}/dataStore")
async def create_dataStore(
    user: str,
    text: str = Form(None),
    audio: UploadFile = File(None),
    image: UploadFile = File(None),
    video: UploadFile = File(None),
    document: UploadFile = File(None),
    metadata: UploadFile = File(None),
    db: Session = Depends(get_db),
    jwt_data: TokenData = Depends(verify_jwt)
):
  
    if user != jwt_data['sub']:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User not authorized",
        )

    if not any([text, audio, image, video, document, metadata]):
        raise HTTPException(status_code=400, detail="At least one argument must be provided")

    try:
        data_id = uuid.uuid4()
        user_id = get_user_id(user, db)
        audio_path = None
        image_path = None
        video_path = None
        document_path = None
        structured_data = None


        if audio:
            file_type = audio.content_type
            if not file_type.startswith('audio/'):
                raise HTTPException(status_code=400, detail="Invalid file type. Only audio files are accepted.")

            audio_path = await build_data_store_path(user, data_id, "audio", audio)
            await save_file_to_path(audio, audio_path)
        
        if image:
            file_type = image.content_type
            if not file_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="Invalid file type. Only image files are accepted.")
            
            image_path = await build_data_store_path(user, data_id, "image", image)
            await save_file_to_path(image, image_path)

        if video:
            file_type = video.content_type
            if not file_type.startswith('video/'):
                raise HTTPException(status_code=400, detail="Invalid file type. Only video files are accepted.")
            
            video_path = await build_data_store_path(user, data_id, "video", video)
            await save_file_to_path(video, video_path)
        
        if document:
            file_type = document.content_type
            if not file_type.startswith('application/pdf'):
                raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are accepted.")
            
            document_path = await build_data_store_path(user, data_id, "document", document)
            await save_file_to_path(document, document_path)

        if metadata:
            file_type = metadata.content_type
            if metadata.content_type not in ['application/json', 'text/csv']:
                raise HTTPException(status_code=400, detail="Invalid file type. Only CSV/JSON files are accepted.")
            
            contents = await metadata.read()
            name, extension = os.path.splittext(metadata.filename)
            if extension == ".json":
                try:
                    structured_data = json.loads(contents)
                except json.JSONDecodeError:
                    raise HTTPException(status_code=400, detail="Invalid JSON file.")
            elif extension == ".csv":
                try:
                    # Decode the content to a string
                    csv_content = contents.decode('utf-8')
                    # Use StringIO to treat the string as a file
                    csv_file = StringIO(csv_content)
                    csv_reader = csv.DictReader(csv_file)
                    rows = list(csv_reader)
                    # Convert to JSON
                    json_data = json.dumps(rows, indent=4)
                    structured_data = json.loads(json_data) 
                except Exception as e:
                    raise HTTPException(status_code=400, detail="Invalid CSV file.")
            
        
        db_dataStorage = models.DataStorage(
            data_id=data_id, 
            user_id=user_id, 
            text_content=text, 
            audio_path=audio_path,
            document_path=document_path,
            metadata = structured_data)
        
        db.add(db_dataStorage)
        db.commit()
        db.refresh(db_dataStorage)


        # Generate access token and create session log
        access_token = create_access_token(data={"sub": user})
        create_session_log(user_id, string_hash(access_token), db)

        return {"user": user, "data_id": data_id, "access_token": access_token, "token_type": "bearer"}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error updating conversation: {str(e)}")

#   LLM Conversation creation
#   Endpoint 5: POST /users/{user}/conversations/
#

@app.post("/users/{user}/conversations/")
async def create_conversation(
    user: str,
    prompt: str = Form(None),
    aud_file: UploadFile = File(None),
    doc_file: UploadFile = File(None),
    db: Session = Depends(get_db),
    jwt_data: TokenData = Depends(verify_jwt)
):
    # Validate user authorization
    if user != jwt_data['sub']:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User not authorized",
        )

    # Validate user input
    if not any([prompt, aud_file, doc_file]):
        raise HTTPException(status_code=400, detail="At least one argument must be provided")

    try:
        # Initialize conversation and message parameters
        convo_uuid = str(uuid.uuid4())
        user_id = get_user_id(user, db)
        message_id = 1
        LLM_prompt = ''
        LLM_response = ''
        aud_file_path = None
        doc_file_path = None
        has_context = False

        # Process prompt input
        if prompt:
            LLM_prompt += f"[Written Context]: {prompt} | "

        # Process audio file input
        if aud_file:
            aud_file_path = await build_conversation_file_path(user, convo_uuid, message_id, "audio", aud_file)
            await save_file_to_path(aud_file, aud_file_path)

            aud_context = get_recognition_response(aud_file_path)
            LLM_prompt += f"[Audio Context]: {aud_context} | "

        # Process document file input
        if doc_file:
            has_context = True
            doc_file_path = await build_conversation_file_path(user, convo_uuid, message_id, "document", doc_file)
            await save_file_to_path(doc_file, doc_file_path)

            context_file_path = await build_conversation_file_path(user, convo_uuid, message_id, "data", "")
            documents = get_document_response(doc_file_path)
            split = split_documents(documents)
            add_to_chroma(context_file_path, split)

            if not LLM_prompt:
                document_context = "\n\n".join([chunk.page_content for chunk in split])
                LLM_prompt = (f"I have a document that provides context for the following conversation. Use the information in "
                            f"the document to answer any questions or generate responses based on the context provided.\n\n"
                            f"Document Context:\n{document_context}\n\nContinue the conversation based on this document context.")
                LLM_response = get_LLM_response(None, prompt=LLM_prompt, max_len=50)
            else:
                LLM_response = get_RAG_response(context_file_path, LLM_prompt)

        # Get LLM response if not already set
        if not LLM_response:
            LLM_response = get_LLM_response(None, prompt=LLM_prompt, max_len=50)

        # Create and save conversation to the database
        db_conversation = models.LLMConversation(
            convo_uuid=convo_uuid,
            has_context=has_context,
            user_id=user_id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            num_messages=message_id
        )
        db.add(db_conversation)
        db.commit()
        db.refresh(db_conversation)

        # Create and save conversation message to the database
        db_conversation_message = models.LLMConversationMessage(
            id=message_id,
            convo_id=convo_uuid,
            user_id=user_id,
            created_at=datetime.utcnow(),
            audio_file_path=aud_file_path,
            document_file_path=doc_file_path,
            message_prompt=LLM_prompt,
            message_response=LLM_response
        )
        db.add(db_conversation_message)
        db.commit()
        db.refresh(db_conversation_message)

        # Generate access token and create session log
        access_token = create_access_token(data={"sub": user})
        create_session_log(user_id, string_hash(access_token), db)

        return {"user": user, "conversation": convo_uuid, "access_token": access_token, "token_type": "bearer"}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error updating conversation: {str(e)}")


#   LLM Conversation message submission
#   Endpoint 6: POST /users/{user}/conversations/{convo_uuid}/messages
#


@app.post("/users/{user}/conversations/{convo_uuid}/messages")
async def create_conversation_message(
    user: str,
    convo_uuid: str,
    prompt: str = Form(None),
    aud_file: UploadFile = File(None),
    doc_file: UploadFile = File(None),
    db: Session = Depends(get_db),
    jwt_data: TokenData = Depends(verify_jwt)
):
    # Validate user authorization
    if user != jwt_data['sub']:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User not authorized",
        )   

    # Validate user input
    if not any([prompt, aud_file, doc_file]):
        raise HTTPException(status_code=400, detail="At least one argument must be provided")

    try:
        # Initialize conversation, message params
        user_id = get_user_id(user, db)
        conversation = get_conversation_from_id(convo_uuid, db)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        has_context = conversation.has_context
        message_id = conversation.num_messages + 1
        LLM_prompt = format_llm_message_history(convo_uuid, db)
        current_prompt = ''
        LLM_response = ''
        aud_file_path = None
        doc_file_path = None

        if prompt:
            current_prompt += "[Written Context]: " + prompt + " | "

        if aud_file:
            audio_path = build_conversation_file_path(user, convo_uuid, message_id, "audio", aud_file.filename)
            await save_file_to_path(aud_file, audio_path)

            aud_context = get_recognition_response(audio_path) 
            current_prompt += "[Audio Context]: " + aud_context + " | "
            aud_file_path = audio_path

        if doc_file:
            doc_file_path = build_conversation_file_path(user, convo_uuid, message_id, "document", doc_file.filename)
            await save_file_to_path(doc_file, doc_file_path)

            context_file_path = build_conversation_file_path(user, convo_uuid, message_id, "data", "")
            documents = get_document_response(doc_file_path)
            split = split_documents(documents)
            add_to_chroma(context_file_path, split)
            has_context = True

            if not current_prompt: 
                document_context = "\n\n".join([chunk.page_content for chunk in split])
                current_prompt = (f"I have a document that provides context for the following conversation. Use the information in "
                            f"the document to answer any questions or generate responses based on the context provided.\n\n"
                            f"Document Context:\n{document_context}\n\nContinue the conversation based on this document context.")
                LLM_prompt += current_prompt
                LLM_response = get_LLM_response(None, prompt=LLM_prompt, max_len=50)
            else:
                LLM_prompt += current_prompt
                LLM_response = get_RAG_response(context_file_path, LLM_prompt)

        if not LLM_response:
            LLM_prompt += current_prompt
            LLM_response = get_LLM_response(None, prompt=LLM_prompt, max_len=50)

        # Update conversation fields
        conversation.num_messages = message_id
        conversation.updated_at = datetime.utcnow()
        conversation.has_context = has_context

        db.commit()
        db.refresh(conversation)

        # Create conversation message
        db_conversation_message = models.LLMConversationMessage(
            id=message_id, 
            convo_id=convo_uuid, 
            user_id=user_id, 
            created_at=datetime.utcnow(), 
            audio_file_path=aud_file_path, 
            document_file_path=doc_file_path, 
            message_prompt=current_prompt, 
            message_response=LLM_response
        )

        db.add(db_conversation_message)
        db.commit()
        db.refresh(db_conversation_message)

        access_token = create_access_token(data={"sub": user})
        create_session_log(user_id, string_hash(access_token), db)
        return {"user": user, "conversation": convo_uuid, "access_token": access_token, "token_type": "bearer"}


    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error updating conversation: {str(e)}")


    
#   Stable Diffusion image generation
#   Endpoint 7: POST /users/{user}/illustrations/
#


@app.post("/users/{user}/illustration/")
async def create_illustration(
    user: str,
    prompt: str,
    db: Session = Depends(get_db),
    jwt_data: TokenData = Depends(verify_jwt) 
):
    # Validate user authorization
    if user != jwt_data.get('sub'):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User not authorized",
        )
    
    # Validate user input
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt must be provided to generate image.")
    
    try:
        illustration_uuid = str(uuid.uuid4())
        user_id = get_user_id(user, db)

        diffusion_path = f'./users/{user}/illustrations/{illustration_uuid}.png'
        generated_image = get_diffusion_response(prompt)
        generated_image.save(diffusion_path)
       
        db_illustration = models.ImageGeneration(
            illustration_uuid=illustration_uuid,
            user_id=user_id,
            illustration_path=diffusion_path
        )
        db.add(db_illustration)
        db.commit()
        db.refresh(db_illustration)

        access_token = create_access_token(data={"sub": user})
        create_session_log(user_id, string_hash(access_token), db)

        return {
            "user": user,
            "illustration": illustration_uuid,
            "access_token": access_token,
            "token_type": "bearer"
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


#   Image Classification
#   Endpoint 8: POST /users/{user}/classifications/
#

@app.post("/users/{user}/classification/")
async def create_classification(
    user: str,
    categories: str = Form(...),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    jwt_data: TokenData = Depends(verify_jwt) 
):
    # Validate user authorization
    if user != jwt_data.get('sub'):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User not authorized",
        )
    
    # Validate user input
    if not (image and categories):
        raise HTTPException(status_code=400, detail="Image and categories must be provided to classify an image.")
    
    try:

        file_type = image.content_type
        if not file_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid file type. Only image files are accepted.")
    

        # need to save image, pass path to src+labels,  create db obj, create jwt,
        classification_uuid = str(uuid.uuid4())
        user_id = get_user_id(user, db)
    

        classification_path = f'./users/{user}/classification/{classification_uuid}.png'
        await save_file_to_path(image, classification_path)
    
        classifications = get_classification_response(classification_path, categories)
       
        db_classification = models.ImageClassification(
            classification_uuid=classification_uuid,
            user_id=user_id,
            source_path=classification_path,
            categories= categories,
            classification=classifications
        )
        db.add(db_classification)
        db.commit()
        db.refresh(db_classification)

        access_token = create_access_token(data={"sub": user})
        create_session_log(user_id, string_hash(access_token), db)

        return {
            "user": user,
            "classification": classification_uuid,
            "access_token": access_token,
            "token_type": "bearer"
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))



#   Image / Object Detection
#   Endpoint 9: POST /users/{user}/detection/
#

@app.post("/users/{user}/detection/")
async def create_detection(
    user: str,
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    jwt_data: TokenData = Depends(verify_jwt) 
):
    # Validate user authorization
    if user != jwt_data.get('sub'):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User not authorized",
        )
    
    if not (image):
        raise HTTPException(status_code=400, detail="Image must be provided to detect objects in an image.")
    
    try:

        file_type = image.content_type
        if not file_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid file type. Only image files are accepted.")
    
        detection_uuid = str(uuid.uuid4())
        user_id = get_user_id(user, db)

        source_path = f'./users/{user}/detection/{detection_uuid}/source.png'
        await save_file_to_path(image, source_path)

        detection_path = f'./users/{user}/detection/{detection_uuid}/result.png'
        detection = get_detection_response(source_path, detection_path)
       
        db_detection = models.ImageDetection(
            detection_uuid=detection_uuid,
            user_id=user_id,
            source_path=source_path,
            detection=detection,
            detection_path=detection_path
        )
        db.add(db_detection)
        db.commit()
        db.refresh(db_detection)

        access_token = create_access_token(data={"sub": user})
        create_session_log(user_id, string_hash(access_token), db)

        return {
            "user": user,
            "detection": detection_uuid,
            "access_token": access_token,
            "token_type": "bearer"
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


#   Speech Recognition and Translation
#   Endpoint 10: POST /users/{user}/recognition/
#

@app.post("/users/{user}/recognition/")
async def create_recognition(
    user: str,
    audio: UploadFile = File(...),
    db: Session = Depends(get_db),
    jwt_data: TokenData = Depends(verify_jwt) 
):
    if user != jwt_data.get('sub'):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User not authorized",
        )
    
    if not (audio):
        raise HTTPException(status_code=400, detail="Audio must be passed to recognize speech.")
    
    try:

        file_type = audio.content_type
        if not file_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Invalid file type. Only audio files are accepted.")
    
        recognition_uuid = str(uuid.uuid4())
        user_id = get_user_id(user, db)

        source_path = f'./users/{user}/recognition/{recognition_uuid}.mp3'
        await save_file_to_path(audio, source_path)

        recognition = get_recognition_response(source_path)
       
        db_recognition = models.VoiceRecognition(
            recognition_uuid=recognition_uuid,
            user_id=user_id,
            audio_path=source_path,
            recognition=recognition
        )
        db.add(db_recognition)
        db.commit()
        db.refresh(db_recognition)

        access_token = create_access_token(data={"sub": user})
        create_session_log(user_id, string_hash(access_token), db)

        return {
            "user": user,
            "recognition": recognition_uuid,
            "access_token": access_token,
            "token_type": "bearer"
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

#   Speech Generation 
#   Endpoint 11: POST /users/{user}/speech/
#

@app.post("/users/{user}/speech/")
async def create_speech(
    user: str,
    prompt: str = Form(...),
    audio: UploadFile = File(...),
    db: Session = Depends(get_db),
    jwt_data: TokenData = Depends(verify_jwt) 
):
    if user != jwt_data.get('sub'):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User not authorized",
        )
    
    if not (audio and prompt):
        raise HTTPException(status_code=400, detail="Voice audio and prompt must be provided to generate prompt speech.")
    
    try:

        file_type = audio.content_type
        if not file_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Invalid file type. Only audio files are accepted.")
    
        speech_uuid = str(uuid.uuid4())
        user_id = get_user_id(user, db)

        source_path = f'./users/{user}/speech/{speech_uuid}/source.mp3'
        await save_file_to_path(audio, source_path)

        speech_path = f'./users/{user}/speech/{speech_uuid}/voice.mp3'
        get_speech_response(source_path, speech_path, prompt)
       
        db_speech = models.VoiceGeneration(
            speech_uuid=speech_uuid,
            user_id=user_id,
            audio=source_path,
            speech_prompt=prompt,
            speech_path=speech_path
        )
        db.add(db_speech)
        db.commit()
        db.refresh(db_speech)

        access_token = create_access_token(data={"sub": user})
        create_session_log(user_id, string_hash(access_token), db)

        return {
            "user": user,
            "speech": speech_uuid,
            "access_token": access_token,
            "token_type": "bearer"
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
