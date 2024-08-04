CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username VARCHAR NOT NULL UNIQUE,
    email VARCHAR NOT NULL UNIQUE,
    hashed_pwd VARCHAR NOT NULL,
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE inquiries (
    id INTEGER PRIMARY KEY,
    resolved BOOLEAN NOT NULL DEFAULT FALSE,
    contact_name VARCHAR NOT NULL,
    contact_email VARCHAR NOT NULL,
    contact_message TEXT NOT NULL
);

CREATE TABLE sessions (
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    jwt VARCHAR NOT NULL,
    start_time TIMESTAMP NOT NULL
);

CREATE TABLE data_storage (
    data_id VARCHAR PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
    text_content VARCHAR,
    audio_path VARCHAR,
    image_path VARCHAR,
    document_path VARCHAR,
    metadata JSON
);

CREATE TABLE conversations (
    convo_uuid VARCHAR PRIMARY KEY,
    has_context BOOLEAN DEFAULT FALSE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
    updated_at TIMESTAMP DEFAULT current_timestamp,
    num_messages INTEGER DEFAULT 0
);

CREATE TABLE conversation_messages (
    id INTEGER PRIMARY KEY,
    convo_id VARCHAR NOT NULL REFERENCES conversations(convo_uuid) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
    audio_file_path VARCHAR,
    document_file_path VARCHAR,
    message_prompt TEXT,
    message_response TEXT
);


CREATE TABLE illustrations (
    illustration_uuid VARCHAR PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
    prompt VARCHAR NOT NULL,
    illustration_path VARCHAR NOT NULL
);

CREATE TABLE detections (
    detection_uuid VARCHAR PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
    source_path VARCHAR NOT NULL,
    detection VARCHAR NOT NULL,
    detection_path VARCHAR NOT NULL
);

CREATE TABLE classifications (
    classification_uuid VARCHAR PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
    source_path VARCHAR NOT NULL,
    categories VARCHAR NOT NULL,
    classification VARCHAR NOT NULL
);

CREATE TABLE speech (
    speech_uuid VARCHAR PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
    audio_path VARCHAR NOT NULL,
    speech_prompt VARCHAR NOT NULL,
    speech_path VARCHAR NOT NULL
);

CREATE TABLE recognitions (
    recognition_uuid VARCHAR PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
    audio_path VARCHAR NOT NULL,
    recognition VARCHAR NOT NULL
);
