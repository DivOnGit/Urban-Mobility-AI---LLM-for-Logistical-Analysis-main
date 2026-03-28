# Urban Mobility AI: LLM for Logistical Analysis

An AI-powered research project that combines Large Language Models (LLMs), retrieval-augmented generation (RAG), and structured analytical tools to study urban transportation patterns using NYC taxi trip data.

This system is designed as a hybrid mobility intelligence assistant: it can answer analytical questions grounded in historical trip records, compute route-level estimates such as fare and distance, and optionally enrich trip planning with real-time traffic and weather signals. The project is intended for research and experimentation in smart mobility, urban informatics, and AI-assisted logistics.

## Overview

Urban transportation systems generate large volumes of operational data, but turning that data into actionable insight often requires both statistical reasoning and natural language interaction. This project explores how LLMs can serve as an interface layer over urban mobility datasets by routing user queries to either:

- a retrieval pipeline for evidence-grounded analytical responses, or
- specialized computational tools for deterministic outputs such as fare, distance, and route estimates.

The project uses NYC taxi trip data as its primary knowledge base and demonstrates a practical architecture for building intelligent transportation assistants for research purposes.

## Research Motivation

The main motivation behind this project is to investigate how language models can support:

- urban logistics analysis,
- demand-aware mobility reasoning,
- natural language querying over transport datasets,
- hybrid decision systems that combine symbolic tools with LAG/RAG pipelines,
- intelligent trip planning augmented with external real-time context.

Rather than relying only on free-form LLM generation, the system uses a hybrid design where analytical questions are answered from retrieved data and computational questions are handled through structured tool execution. This reduces hallucination risk and improves interpretability.

## Key Features

- Hybrid query handling with LLM-based intent routing
- Retrieval-Augmented Generation over NYC taxi trip records
- Tool-based computation for distance, fare, route, and trip planning
- Zone and borough resolution for taxi location identifiers
- Historical route statistics derived from trip-level data
- Optional real-time traffic integration using Google Distance Matrix API
- Optional weather-aware trip adjustment using OpenWeatherMap API
- Modular Python architecture for experimentation and extension

## System Architecture

The project follows a layered architecture:

1. **User Query Input**
   - A user submits a natural language question such as:
   - `How far is Midtown from JFK Airport?`
   - `How much is a taxi from Chelsea to LaGuardia?`
   - `Plan my trip from SoHo to Times Square.`

2. **LLM-Based Router**
   - The router classifies the query into one of three categories:
   - `ANALYTICAL`
   - `COMPUTATIONAL`
   - `HYBRID`

3. **Execution Path**
   - `ANALYTICAL`: uses vector retrieval + RAG over taxi dataset documents
   - `COMPUTATIONAL`: invokes a structured tool directly
   - `HYBRID`: calls a tool first, then supplements the output with RAG explanation

4. **Response Generation**
   - Tool outputs are summarized for readability
   - RAG answers are constrained to retrieved evidence
   - Final responses are returned in natural language

## Project Modules

### Core Files

- `main.py`  
  Entry point for interactive or one-shot query execution

- `ask.py`  
  Main orchestration logic for routing, tool execution, and RAG fallback

- `llm_router.py`  
  LLM-based intent classification and tool selection

- `llm_layer.py`  
  Handles LLM prompting for reasoning and summarization

- `retriever.py`  
  Retrieves relevant documents from the vector database

- `vector_store.py`  
  Builds and persists the Chroma vector database from taxi records

- `tool_registry.py`  
  Registers and validates all available computational tools

### Tooling Layer

- `tools/distance_tool.py`  
  Estimates route distance using historical trip data

- `tools/fare_tool.py`  
  Estimates route fare using historical median pricing

- `tools/route_optimizer.py`  
  Computes average route duration and fare for optimization-oriented queries

- `tools/urban_trip_planner.py`  
  Combines historical estimates with traffic and weather context

- `tools/zone_resolver.py`  
  Resolves fuzzy user zone names to valid NYC taxi zones

### External Services

- `external_services/traffic_service.py`  
  Integrates Google Distance Matrix API for traffic-aware travel estimates

- `external_services/weather_service.py`  
  Integrates OpenWeatherMap API for weather-based trip adjustments

## Dataset

The system uses NYC taxi trip data stored in:

```bash
data/merged_taxi_data.csv
