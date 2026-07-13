# Urban Mobility AI - Gemma-2-9B-IT

Production-style implementation of the **Urban Mobility AI: An LLM-Based Framework for Explainable and Actionable Transportation Intelligence in Smart Cities** pipeline using `google/gemma-2-9b-it`.

This repository is designed for execution by a user who has access to the required model weights and the NYC Taxi TLC CSV data. It does not include model weights, API keys, private credentials, or fabricated benchmark outputs.

## Architecture

User query -> Intent detection -> Retriever -> FAISS vector search -> Relevant context -> LLM reasoning -> LangChain tools -> Conversation memory -> Final response.

Core capabilities:

- NYC Taxi CSV loading and preprocessing with pandas
- document chunking for trip records
- Sentence Transformers embeddings
- FAISS vector index creation and semantic retrieval
- LangGraph orchestration
- HuggingFace Transformers model loading
- LangChain tools for distance, demand, pattern, trip, time, and location analysis
- Streamlit chat interface
- CLI ingestion and chat flow

## Folder Structure

```text
.
|-- app.py
|-- config.py
|-- dataset_loader.py
|-- embeddings.py
|-- evaluation.py
|-- llm.py
|-- main.py
|-- memory.py
|-- prompts.py
|-- query_router.py
|-- rag.py
|-- retriever.py
|-- sample_queries.py
|-- tools.py
|-- utils.py
|-- vector_store.py
|-- data/
|-- vector_db/
|-- prompts/
|-- results/
`-- images/
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

On Linux, install the CUDA-compatible PyTorch build recommended for your GPU before installing the rest of the requirements.

## Model Setup

Default model:

```text
google/gemma-2-9b-it
```

If the model is gated, request access on HuggingFace and authenticate:

```bash
huggingface-cli login
```

If your institution serves the same model from an internal endpoint or local path, set `MODEL_NAME` in `.env` to that path or model identifier.

## Dataset

Place a TLC-compatible NYC Taxi CSV at:

```text
data/nyc_taxi.csv
```

Or set:

```text
DATA_PATH=/path/to/your/tlc_file.csv
```

The loader expects common TLC columns such as pickup/dropoff timestamps, trip distance, passenger count, fare or total amount, and location IDs when available.

## Run

Build the vector index:

```bash
python main.py --build-index
```

Start CLI chat:

```bash
python main.py --chat
```

Start Streamlit:

```bash
streamlit run app.py
```

## Hardware

24 GB VRAM recommended for 4-bit inference; 40+ GB for higher precision.

The project supports 4-bit quantization through BitsAndBytes when available. Large 70B-class models usually require multi-GPU inference, high-memory accelerators, or an optimized serving stack.

## Expected Outputs

- Grounded answer to the user query
- Retrieved NYC taxi context
- Tool-derived statistics when applicable
- Caveats when data is missing or insufficient
- Persisted FAISS index under `vector_db/`
- Evaluation responses under `results/`

## Screenshot Placeholders

Place interface and experiment screenshots in `images/`, for example:

- `images/streamlit_chat.png`
- `images/retrieval_trace.png`
- `images/evaluation_summary.png`

## Future Work

- Add borough lookup joins using TLC taxi zone shapefiles
- Add model-serving adapters for vLLM or Text Generation Inference
- Add structured evaluation metrics after running the approved benchmark suite
- Add geospatial visualization of pickup and dropoff demand

