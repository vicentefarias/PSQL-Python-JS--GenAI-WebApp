# SQL-Python-JS Generative Artificial Intelligence Web Application 

## Overview
- This project implements a postgreSQL relational database, FastAPI REST backend server, and reactJS frontend client for a generative AI web application. Users can upload and store data, choose from transformer models to generate text, images, or audio, and run inference on other ML tasks such as retrieval augmented generation using LLMs to perform document question and answering, object detection using detection transformer, or speech recognition & translation.

 - Each project component (database, backend server, frontend client) can be setup and run independently, but are meant to be integrated as part of a full stack web application. 

## Key Features
- Database: 
The database design emphasizes user-centric data management, ensuring each user has a unique identity and securely stored credentials. It supports comprehensive tracking of user interactions and sessions, enabling detailed session logs and data storage with robust relational mappings. The schema incorporates modularity, with distinct tables for various AI-driven functionalities like image generation, detection, classification, and voice-related processes. Each table includes foreign key constraints to maintain referential integrity and facilitate cascading deletions, ensuring data consistency and streamlined user-specific data handling. The use of UUIDs for primary keys in most tables enhances security and uniqueness across records.

- Backend:
The FastAPI REST backend design focuses on a robust and flexible approach to handling various types of data and user interactions. It includes endpoints for managing conversations, processing text and audio data, generating images, and performing classification and detection tasks. Each endpoint ensures user authorization, validates inputs, and processes files efficiently while maintaining error handling and database integrity. Overall, the design prioritizes security, user-specific data management, and scalability, with a clear emphasis on handling multimedia content and integrating with AI models for advanced functionalities.

- Frontend:

## Project Setup: Database

## Project Setup: REST API Backend

## Project Setup: Frontend Client