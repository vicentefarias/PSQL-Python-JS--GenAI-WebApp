# SQL-Python-JS Generative Artificial Intelligence Web Application 

## Overview
- This project implements a postgreSQL relational database, FastAPI REST backend server, and reactJS frontend client for a generative AI web application. Users can upload and store different abstract data types, choose from transformer models to generate text, images, or audio, and run inference on other ML tasks such as retrieval augmented generation using LLMs to perform document question and answering, object detection using detection transformer, or speech recognition & translation.

 - Each project component (database, backend server, frontend client) can be setup and run independently, but are meant to be integrated as part of a full stack web application. 

## Key Features
- Database: 
The database design emphasizes user-centric data management, ensuring each user has a unique identity and securely stored credentials. It supports comprehensive tracking of user interactions and sessions, enabling detailed session logs and data storage with robust relational mappings. The schema incorporates modularity, with distinct tables for various AI-driven functionalities like image generation, detection, classification, and voice-related processes. Each table includes foreign key constraints to maintain referential integrity and facilitate cascading deletions, ensuring data consistency and streamlined user-specific data handling. The use of UUIDs for primary keys in most tables enhances security and uniqueness across records.

- Backend:
The FastAPI REST backend design focuses on a robust and flexible approach to handling various types of data and user interactions. It includes endpoints for managing conversations, processing text and audio data, generating images, and performing classification and detection tasks. Each endpoint ensures user authorization, validates inputs, and processes files efficiently while maintaining error handling and database integrity. The design prioritizes security by accepting requests strictly from the reactJS frontend server, user-specific data management, and scalability, with a clear emphasis on handling multimedia content and integrating with AI models for advanced functionalities.

- Frontend:
The frontend ReactJS components are designed with a focus on modularity, reusability, and user-centric interactions. Each component serves a distinct function, from authentication and data management to advanced AI capabilities such as language model inference, image generation, and speech processing. The design emphasizes a clean and responsive UI, ensuring seamless user experiences across various devices. State management and API integration are handled efficiently to maintain robust and dynamic data interactions, while error handling and feedback mechanisms are incorporated to enhance reliability and user trust.

## Project Setup: Database

- In a postgreSQL environment, run the postgreSQL/tables.sql file to create the tables

## Project Setup: REST API Backend

- To run the backend server, download the Python FastAPI backend server and execute the main script with the following command. A number of dependencies are required, specified in requirements.txt. A GPU runtime is recommended to run the endpoints associated with ML model inference.


cd ./PythonFastAPI
python3 -m pip install requirements.txt
python3 -m uvicorn main:app --reload

## Project Setup: Frontend Client

- To run the frontend client, download the reactJS project components. Dependencies 