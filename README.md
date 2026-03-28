<div align="center">

<sub>RESEARCH PROJECT</sub>

# Urban Mobility AI
### *LLM for Logistical Analysis*

A hybrid intelligence assistant combining Large Language Models, Retrieval-Augmented Generation, and structured analytical tools to study urban transportation patterns using NYC taxi trip data.

![LLM Routing](https://img.shields.io/badge/LLM_Routing-0d1a3a?style=flat-square&color=0d1a3a&labelColor=4f7cff&label=)
![RAG Pipeline](https://img.shields.io/badge/RAG_Pipeline-0a2021?style=flat-square&color=0a2021&labelColor=38b2ac&label=)
![NYC Taxi Data](https://img.shields.io/badge/NYC_Taxi_Data-1f1500?style=flat-square&color=1f1500&labelColor=f6a623&label=)
![Real-time Traffic](https://img.shields.io/badge/Real--time_Traffic-0d1a3a?style=flat-square&color=0d1a3a&labelColor=4f7cff&label=)
![Weather Signals](https://img.shields.io/badge/Weather_Signals-0a2021?style=flat-square&color=0a2021&labelColor=38b2ac&label=)

</div>

---

## `01` Research Motivation

Urban transportation systems generate enormous volumes of operational data, yet turning that data into actionable insight typically requires both statistical reasoning and natural language interaction. This project investigates how language models can serve as an interface layer over urban mobility datasets.

Rather than relying on free-form LLM generation, the system uses a hybrid design: analytical questions are answered from retrieved data while computational questions are routed through structured tool execution. This reduces hallucination risk and improves interpretability.

The main areas of investigation:

- Urban logistics analysis
- Demand-aware mobility reasoning
- Natural language querying over transport datasets
- Hybrid decision systems combining symbolic tools with RAG pipelines
- Intelligent trip planning augmented with real-time external context

---

## `02` Key Features

<table>
<tr>
<td>✦ Hybrid query handling with LLM-based intent routing</td>
<td>✦ Retrieval-augmented generation over NYC taxi trip records</td>
</tr>
<tr>
<td>✦ Tool-based computation for distance, fare, route, and trip planning</td>
<td>✦ Zone and borough resolution for taxi location identifiers</td>
</tr>
<tr>
<td>✦ Historical route statistics derived from trip-level data</td>
<td>✦ Optional real-time traffic via Google Distance Matrix API</td>
</tr>
<tr>
<td>✦ Weather-aware trip adjustment via OpenWeatherMap API</td>
<td>✦ Modular Python architecture for experimentation and extension</td>
</tr>
</table>

---

## `03` System Architecture

```
User Query Input
      │
      ▼
 LLM Router
      │
      ├─── ANALYTICAL ──────► Vector Retrieval + RAG
      │                              │
      ├─── COMPUTATIONAL ───► Structured Tool Execution
      │                              │
      └─── HYBRID ──────────► Tool First → RAG Supplement
                                     │
                                     ▼
                          Natural Language Response
```

### Query routing

| Type | Execution Path | Example Query |
|------|---------------|---------------|
| `ANALYTICAL` | Vector retrieval + RAG over taxi dataset | *"What zones have the highest demand on weekends?"* |
| `COMPUTATIONAL` | Direct tool invocation | *"How much is a taxi from Chelsea to LaGuardia?"* |
| `HYBRID` | Tool call → RAG explanation | *"Plan my trip from SoHo to Times Square."* |

### Pipeline steps

**01 — User Query Input**
Natural language questions such as `How far is Midtown from JFK?` or `Plan my trip from SoHo to Times Square.`

**02 — LLM-Based Router**
Classifies the query into one of three categories: `ANALYTICAL`, `COMPUTATIONAL`, or `HYBRID`.

**03 — Execution Path**
Routes to the appropriate pipeline based on classification.

**04 — Response Generation**
Tool outputs are summarized; RAG answers are constrained to retrieved evidence. Final responses are returned in natural language.

---

## `04` Project Modules

### Core files

| File | Description |
|------|-------------|
| `main.py` | Entry point for interactive or one-shot query execution |
| `ask.py` | Main orchestration: routing, tool execution, and RAG fallback |
| `llm_router.py` | LLM-based intent classification and tool selection |
| `llm_layer.py` | LLM prompting for reasoning and summarization |
| `retriever.py` | Retrieves relevant documents from the vector database |
| `vector_store.py` | Builds and persists the Chroma vector database from taxi records |
| `tool_registry.py` | Registers and validates all available computational tools |

### Tooling layer

| File | Description |
|------|-------------|
| `tools/distance_tool.py` | Estimates route distance using historical trip data |
| `tools/fare_tool.py` | Estimates route fare using historical median pricing |
| `tools/route_optimizer.py` | Computes average route duration and fare |
| `tools/urban_trip_planner.py` | Combines historical estimates with traffic and weather context |
| `tools/zone_resolver.py` | Resolves fuzzy user zone names to valid NYC taxi zones |

### External services

| File | Description |
|------|-------------|
| `external_services/traffic_service.py` | Google Distance Matrix API for traffic-aware travel estimates |
| `external_services/weather_service.py` | OpenWeatherMap API for weather-based trip adjustments |

---

## `05` Dataset

The system uses NYC taxi trip data as its primary knowledge base, powering both the RAG retrieval pipeline and the statistical estimates returned by computational tools.

```bash
data/merged_taxi_data.csv
```

---

<div align="center">
<sub>Urban Mobility AI · Research Project · NYC Taxi Trip Data</sub>
</div>
