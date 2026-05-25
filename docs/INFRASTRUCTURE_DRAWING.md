## TamuBot — Infrastructure & Evaluation Diagrams

Companion to `docs/MAIN_DRAWING.md`. Plain-label flavor — designed to paste cleanly into Excalidraw's mermaid importer (no `<br/>` tags, minimal styling).

1. **Hardware & Container topology** — from the user's browser down to the silicon, splitting *local* CPU/disk/network from *remote* GPU/storage.
2. **Evaluation & Observability pipeline** — how `run_probe`, `run_benchmark`, `eval_chunking`, and `run_eval` wrap the same LangGraph pipeline with tracing + RAGAS scoring.

---

## 1. Hardware & Container Topology

Local box is CPU-only — Python orchestration, disk I/O, and network egress. All AI compute runs on remote provider GPUs over HTTPS. The api-proxy container is the only path out to TAMU + Voyage so we can rate-limit and log.

```mermaid
---
title: TamuBot — Hardware & Container Topology
---
%%{init: {'flowchart': {'curve': 'step'}}}%%
graph LR

    User([User Browser - localhost 8501])

    subgraph Host [Host Machine - Windows 11 WSL2 or macOS]
        direction TB

        subgraph HW [Local Hardware - CPU only]
            direction LR
            CPU[CPU - x86-64 cores running Python and LangGraph]
            RAM[RAM - Streamlit session and retrieved chunks]
            DISK[(Disk SSD - tamu_data processed evals logs env)]
            NIC[NIC - HTTPS egress and port 8501 ingress]
        end

        subgraph DockerEngine [Docker Engine - docker-compose]
            direction TB

            subgraph DevC [Container tamubot-dev-1]
                direction TB
                Streamlit[Streamlit app on port 8501]
                Pipeline[LangGraph pipeline - router retrieval generator]
                Playwright[Playwright and Chromium - ingestion scrapers]
                ClaudeCLI[Claude Code CLI - dev sessions]
            end

            subgraph ProxyC [Container tamubot-api-proxy-1]
                direction TB
                Proxy[API proxy on port 8080 - rate limit and logging]
            end

            Vol[(Bind mount workspace - tamu_data logs src)]
        end
    end

    subgraph Remote [Remote Services - where the heavy compute lives]
        direction TB

        subgraph TAMUCluster [TAMU LLM gateway]
            TAMU_GPU[GPU cluster - chat-api.tamu.ai - generator router critic]
        end

        subgraph VoyageCluster [Voyage AI]
            Voyage_GPU[GPU inference - voyage-3 embeddings and rerank-2]
        end

        subgraph AtlasCluster [MongoDB Atlas]
            Atlas_Store[(Managed cluster - vector index text index chunks)]
        end

        subgraph GeminiCluster [Google Gemini]
            Gemini_GPU[TPU GPU inference - ingestion OCR and structure extraction]
        end

        Mem0Cloud[(mem0 Cloud - long-term memory per session)]
        LangfuseCloud[(Langfuse Cloud - traces and scores)]
    end

    User --> Streamlit
    Streamlit --> Pipeline

    DevC -.runs on.-> CPU
    DevC -.RSS.-> RAM
    ProxyC -.runs on.-> CPU
    Vol -.backed by.-> DISK
    DevC --- Vol
    ProxyC --- Vol

    Pipeline ==> Proxy

    Proxy --> NIC
    Pipeline --> NIC
    Playwright --> NIC

    NIC ==> TAMU_GPU
    NIC ==> Voyage_GPU
    NIC ==> Atlas_Store
    NIC ==> Mem0Cloud
    NIC ==> LangfuseCloud
    NIC ==> Gemini_GPU

    classDef hw fill:#eceff1,stroke:#455a64,stroke-width:2px
    classDef container fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef gpu fill:#ffccbc,stroke:#bf360c,stroke-width:3px
    classDef store fill:#c5e1a5,stroke:#33691e,stroke-width:2px
    classDef cloud fill:#fff59d,stroke:#fbc02d,stroke-width:2px

    class CPU,RAM,DISK,NIC hw
    class Streamlit,Pipeline,Playwright,ClaudeCLI,Proxy,Vol container
    class TAMU_GPU,Voyage_GPU,Gemini_GPU gpu
    class Atlas_Store store
    class Mem0Cloud,LangfuseCloud cloud

    style Host fill:#f5f5f5,stroke:#37474f,stroke-width:3px
    style Remote fill:#fafafa,stroke:#e65100,stroke-width:3px,stroke-dasharray: 6 4
    style HW fill:#fff,stroke:#607d8b,stroke-dasharray: 4 2
    style DockerEngine fill:#fff,stroke:#0d47a1,stroke-dasharray: 4 2
    style DevC fill:#bbdefb,stroke:#1565c0
    style ProxyC fill:#bbdefb,stroke:#1565c0
    style TAMUCluster fill:#ffe0b2,stroke:#e65100
    style VoyageCluster fill:#ffe0b2,stroke:#e65100
    style AtlasCluster fill:#dcedc8,stroke:#33691e
    style GeminiCluster fill:#ffe0b2,stroke:#e65100
```

**Reading guide.**
- **Bold arrows (`==>`)** are the hot path of a user query: Streamlit → pipeline → proxy → TAMU/Voyage GPUs, plus a direct hop to Atlas.
- **Local hardware** does *zero* model inference. CPU runs Python state + LangGraph, RAM holds the session, the SSD holds processed syllabi + golden sets + logs, the NIC pushes every model call out.
- **Remote hardware** is where the GPUs live: TAMU hosts the generator/router/critic, Voyage hosts embeddings + reranker, Gemini handles ingestion OCR, Atlas serves the indexes.

---

## 2. Evaluation & Observability Pipeline

Same LangGraph pipeline reused by four runners. Each runner picks a preset `ObservabilityConfig` that decides trace name, tags, and which `EvalBlock`s fire after the pipeline returns. RAGAS blocks call TAMU (critic LLM) and Voyage (critic embeddings), then post scores back to Langfuse alongside the original trace.

```mermaid
---
title: TamuBot — Evaluation & Observability Pipeline
---
%%{init: {'flowchart': {'curve': 'step'}}}%%
graph LR

    subgraph Inputs [Inputs]
        direction TB
        Golden[(Golden sets - tamu_data evals golden_sets xlsx)]
        AdHoc[Ad-hoc query - query or test-ids]
        Prod[Production user via Streamlit]
    end

    subgraph Runners [Runners - rag observability presets]
        direction TB
        Probe[run_probe.py - probe_config - async RAGAS]
        Bench[run_benchmark.py - benchmark_config - sync RAGAS]
        Chunk[eval_chunking.py - chunking_config - retrieval only]
        Eval[run_eval.py - eval_config - unified and configurable]
        App[app.py streamlit - prod_config - traces only no evals]
    end

    subgraph PipelineSubgraph [LangGraph pipeline - inside trace_context]
        direction TB
        TraceStart{{create_trace - tamubot request probe or benchmark}}
        Router[node router]
        Retrieval[node retrieval - embed search hybrid semantic rerank]
        Generator[node generator and generator comparison]
        TraceEnd{{finalize_trace - lf flush}}

        TraceStart --> Router --> Retrieval --> Generator --> TraceEnd
    end

    subgraph EvalEngine [Evaluation engine - rag observability evals py]
        direction TB
        BuildInputs[Build EvalInputs - question contexts answer reference]
        RunEvals{{run_evals - async or sync - retry once on failure}}

        subgraph Blocks [EvalBlocks - ragas_blocks py]
            direction LR
            Faith[FaithfulnessBlock - answer grounded in contexts]
            Rel[AnswerRelevancyBlock - answer addresses question]
            Prec[ContextPrecisionBlock - needs reference]
            Recall[ContextRecallBlock - needs reference]
        end

        Critics[Critic singletons - TAMU LLM and Voyage embeddings]
    end

    subgraph Outputs [Outputs]
        direction TB
        LF[(Langfuse Cloud - traces span tree scores 0 to 1 or -1)]
        Excel[(Excel reports - tamu_data evals reports benchmark xlsx)]
        MD[(Markdown reports - tamu_data evals reports benchmark md)]
        Stdout[Console output - per query summary]
    end

    AdHoc --> Probe
    Golden --> Bench
    Golden --> Chunk
    Golden --> Eval
    Prod --> App

    Probe --> TraceStart
    Bench --> TraceStart
    Chunk --> TraceStart
    Eval --> TraceStart
    App --> TraceStart

    Router -. span .-> LF
    Retrieval -. spans .-> LF
    Generator -. generation .-> LF
    TraceEnd -. flush .-> LF

    TraceEnd --> BuildInputs
    BuildInputs --> RunEvals
    RunEvals --> Faith
    RunEvals --> Rel
    RunEvals --> Prec
    RunEvals --> Recall

    Faith -.uses.-> Critics
    Rel -.uses.-> Critics
    Prec -.uses.-> Critics
    Recall -.uses.-> Critics

    Faith --> LF
    Rel --> LF
    Prec --> LF
    Recall --> LF

    Critics ==> TAMU_API([TAMU LLM])
    Critics ==> Voyage_API([Voyage AI])

    Bench --> Excel
    Bench --> MD
    Eval --> Excel
    Eval --> MD
    Probe --> Stdout

    App -.no eval blocks.-> LF

    classDef runner fill:#e1bee7,stroke:#4a148c,stroke-width:2px
    classDef pipe fill:#fff,stroke:#333,stroke-width:2px
    classDef block fill:#b3e5fc,stroke:#01579b,stroke-width:2px
    classDef sink fill:#fff59d,stroke:#fbc02d,stroke-width:2px
    classDef src fill:#c5e1a5,stroke:#33691e,stroke-width:2px
    classDef ext fill:#ffab91,stroke:#bf360c,stroke-width:2px

    class Probe,Bench,Chunk,Eval,App runner
    class Router,Retrieval,Generator,TraceStart,TraceEnd,BuildInputs,RunEvals pipe
    class Faith,Rel,Prec,Recall,Critics block
    class LF,Excel,MD,Stdout sink
    class Golden,AdHoc,Prod src
    class TAMU_API,Voyage_API ext

    style Inputs fill:#f1f8e9,stroke:#33691e,stroke-dasharray: 4 2
    style Runners fill:#f3e5f5,stroke:#4a148c,stroke-dasharray: 4 2
    style PipelineSubgraph fill:#eceff1,stroke:#37474f,stroke-dasharray: 4 2
    style EvalEngine fill:#e3f2fd,stroke:#0d47a1,stroke-dasharray: 4 2
    style Blocks fill:#e1f5fe,stroke:#01579b,stroke-dasharray: 4 2
    style Outputs fill:#fffde7,stroke:#f57f17,stroke-dasharray: 4 2
```

**Reading guide.**
- **Runners are interchangeable wrappers.** They all build the same trace and run the same LangGraph; the only differences are the preset `ObservabilityConfig` (trace name, tags, which `eval_blocks` fire, async vs sync, generator on/off).
- **Tracing is live.** Spans (`node.router`, `node.retrieval.*`, `node.generator`) stream to Langfuse during pipeline execution; `finalize_trace` flushes on completion.
- **Eval blocks run *after* the pipeline returns.** `EvalInputs` is assembled from final state, then `run_evals` dispatches the registered RAGAS blocks. Failures get a `-1` score with metadata so the trace is never silently dropped.
- **Critic LLM / embeddings are reused singletons** — TAMU gateway for reasoning, Voyage for embedding-based metrics. They make their own HTTPS calls during scoring, separate from the pipeline call.
- **Production (`prod_config` in app.py) ships traces only — no eval blocks fire on user queries.** RAGAS scoring is reserved for probe/benchmark/chunking/eval runs.
