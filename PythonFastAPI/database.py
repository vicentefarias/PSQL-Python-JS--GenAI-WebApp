from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base

# postgresql://[username]:[password]@[host]:[port]/[database]
URL_DATABASE = "postgresql://postgres:postgres@localhost:5432/genai"

engine= create_engine(URL_DATABASE)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
