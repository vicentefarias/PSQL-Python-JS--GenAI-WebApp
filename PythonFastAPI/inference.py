from transformers import DetrImageProcessor, DetrForObjectDetection, CLIPProcessor, CLIPModel, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from diffusers import AutoPipelineForText2Image 
from PIL import Image
from TTS.api import TTS
import transformers
import torch
import cv2
import numpy as np
import pypdf

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16


llama_model = "meta-llama/Meta-Llama-3-8b-Instruct"
llm_pipeline = transformers.pipeline(
    "text-generation",
    model=llama_model,
    model_kwargs={"torch_dtype": torch_dtype},
    device_map="auto",
)

stable_diffusion_model =  "stabilityai/sdxl-turbo"
diffusion_pipeline = AutoPipelineForText2Image.from_pretrained(stable_diffusion_model, torch_dtype=torch.float16, variant="fp16")
diffusion_pipeline.to("cuda")

detr_model = "facebook/detr-resnet-50"
detection_processor = DetrImageProcessor.from_pretrained(detr_model)
detection_pipeline = DetrForObjectDetection.from_pretrained(detr_model)

vit_model = "openai/clip-vit-large-patch14"
classification_pipeline = CLIPModel.from_pretrained(vit_model)
classification_processor = CLIPProcessor.from_pretrained(vit_model)

whisper_model = "openai/whisper-large-v3"
recognition_preprocessor = AutoModelForSpeechSeq2Seq.from_pretrained(
    whisper_model, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
recognition_preprocessor.to(device)
recognition_processor = AutoProcessor.from_pretrained(whisper_model)
recognition_pipeline = pipeline(
    "automatic-speech-recognition",
    model=recognition_preprocessor,
    tokenizer=recognition_processor.tokenizer,
    feature_extractor=recognition_processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device
)

tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts_model.to("cuda")

## Large Language Model 

def get_LLM_response(messages: list[str], prompt: str, max_len: int) -> str:
    if not messages:
        messages = [{"role": "system", "content" : "You are a chatbot who assists with productivity and coding tasks."},]
    messages.append({"role":"user", "content": prompt})

    terminators = [
    llm_pipeline.tokenizer.eos_token_id,
    llm_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    outputs = llm_pipeline(
    messages,
    max_new_tokens=max_len,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    )
    print(outputs[0]["generated_text"][-1]["content"])
    
    torch.cuda.empty_cache()  # Clear the GPU cache to free up memory
    
    return outputs[0]["generated_text"][-1]["content"]


## Large Language Model  Retrieval Augmented Generation

def get_document_response(src_path: str):
    loader = PyPDFLoader(src_path)
    pages = loader.load()
    return pages

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=10,
        length_function=len,
    )
    return text_splitter.split_documents(documents)

def get_embedding_function():
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embedding_function

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

def add_to_chroma(data_directory, chunks: list[Document]):
    db = Chroma(persist_directory=data_directory, embedding_function=get_embedding_function())

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()

def get_RAG_response(data_directory: str, query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=data_directory, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}

Answer the following question based on the above context: {question}
    """
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    messages = [
        {"role": "system", "content": context_text},
        {"role": "user", "content": query_text},
    ]

    outputs = llm_pipeline(
        messages,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    
    torch.cuda.empty_cache()  # Clear the GPU cache to free up memory
    
    return outputs[0]["generated_text"][-1]




## Image Generation / Stable Diffusion

def get_diffusion_response(prompt: str):
    return diffusion_pipeline(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

## Image / Object Detection

def get_detection_response(str_path: str, dst_path: str):
    try:
        image = Image.open(str_path)
        inputs = detection_processor(images=image, return_tensors="pt")
        outputs = detection_pipeline(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = detection_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
        result = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
        detection = ""
        for i, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
            box = box.tolist()
            x_min = int(box[0])
            y_min = int(box[1])
            x_max = int(box[2])
            y_max = int(box[3])
            w = y_max - y_min
            h = x_max - x_min
            color_idx = i % len(color_list)
            color = color_list[color_idx]
            x = cv2.rectangle(result, (x_min, y_min), (x_min+w, y_min+h), color, 4)
            label_text = f"{detection_pipeline.config.id2label[label.item()]}: {score:.3f}"
            x = cv2.putText(result, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
            detection = detection + f"Detected {detection_pipeline.config.id2label[label.item()]} with confidence "
            detection = detection + f"{round(score.item(), 3)} at location {box}\n"
        cv2.imwrite(dst_path, result)
        torch.cuda.empty_cache()
        return detection
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

## Image Classification

def get_classification_response(src_path: str, labels: str):
    try:
        image = Image.open(src_path)
        label_list = labels.split(",")
        inputs = classification_processor(text=label_list, images=image, return_tensors="pt", padding=True)
        outputs = classification_pipeline(**inputs)
        logits_per_image = outputs.logits_per_image 
        probs = logits_per_image.softmax(dim=1)   
        classifications = ""
        for i,j in zip(label_list, probs.tolist()[0]):
            classifications += i +" label confidence: " + str(round(j, 3)) + "\n"
        torch.cuda.empty_cache()
        return classifications
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


## Speech Language Recognition and Translation

def get_recognition_response(src_path: str):
    try:
        result = recognition_pipeline(src_path, generate_kwargs={"language": "english"})
        torch.cuda.empty_cache()  # Clear the GPU cache to free up memory
        return result['text']
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

## Speech Generation


def get_speech_response(src_path: str, dst_path: str, text: str):
    try:
        result = tts_model.tts_to_file(
            text=text,
            file_path=dst_path,
            speaker_wav=src_path,
            language="en"
        )
        torch.cuda.empty_cache()
        return dst_path
    except Exception as e:
        print(f"An error occurred: {e}")
        return None