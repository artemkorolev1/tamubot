## LangGraph Pipeline

The `build_graph_with_memory()` state machine (`rag/graph/builder.py`). Outer arc = Streamlit session loop. Center cluster = underlying infrastructure. Recursive path (right branch) activates only for multi-course comparison queries. Advisory path (left branch) activates for `PLANNING` / `CAREER` intents — the router hands off to the advisory orchestrator (`advisory/graph.py`) which collects student context via an SCP form, then invokes a headless RAG graph (no router).

graph TD

    %% Global Observability (Side Node)

    Langfuse([Langfuse])



    %% Top section

    Mem0_T([mem0 Cloud]) <-.-> H_Inject

    Start((Start)) --> H_Inject[**HISTORY_INJECT**]

    

    %% Routing section

    TAMU_T([TAMU LLM]) <-.-> Router

    H_Inject --> Router{**ROUTER**}

    

    %% Main Paths

    Router -->|out_of_scope| Canned[**CANNED_RESPONSE**]

    Router -->|retrieval| Retrieval

    Router -->|"advisory (PLANNING / CAREER)"| SCP_Form

    

    %% Advisory Orchestrator Subgraph

    subgraph Advisory_Orchestrator [Advisory Orchestrator]

        direction TB

        SCP_Form[/**SCP FORM**<br/>Streamlit sidebar/] --> PolicyTool[**POLICY_TOOL**]

        PolicyTool --> RagTool[**RAG_TOOL**]

        RagTool --> ResponsePass[**RESPONSE_PASSTHROUGH**]

    end

    

    %% RAG Tool calls headless graph

    RagTool -.->|"headless RAG<br/>(no router)"| Retrieval

    

    %% Synthesis Engine Subgraph

    subgraph Synthesis_Engine [Synthesis Engine]

        direction TB

        Retrieval[**RETRIEVAL**] --> Generator[**GENERATOR**]

        Generator ==>|"recursive loop (updated state)"| Retrieval

    end

    

    %% External Retrieval Tools

    Atlas([Mongo DB Atlas]) <-.-> Retrieval

    Voyage([Voyage AI]) <-.-> Retrieval

    

    %% Generator Tool

    Generator <-.-> TAMU_B([TAMU LLM])

    

    %% Closing section

    Canned --> H_Update[**HISTORY_UPDATE**]

    Generator --> H_Update

    ResponsePass --> H_Update

    H_Update <-.-> Mem0_B([mem0 Cloud])

    H_Update --> End((End))



    %% Langfuse Observability Traces (Thin Dashed)

    Langfuse -.- H_Inject

    Langfuse -.- Router

    Langfuse -.- Retrieval

    Langfuse -.- Generator

    Langfuse -.- H_Update

    Langfuse -.- RagTool



    %% Styling for Bold/Bigger appearance

    style Router fill:#f48fb1,stroke:#880e4f,stroke-width:4px

    style H_Inject fill:#fff,stroke:#333,stroke-width:4px

    style Retrieval fill:#fff,stroke:#333,stroke-width:4px

    style Generator fill:#fff,stroke:#333,stroke-width:4px

    style Canned fill:#fff,stroke:#333,stroke-width:4px

    style H_Update fill:#fff,stroke:#333,stroke-width:4px

    

    %% Advisory Orchestrator styling

    style Advisory_Orchestrator fill:#e3f2fd,stroke:#1565c0,stroke-dasharray: 5 5

    style SCP_Form fill:#bbdefb,stroke:#1565c0,stroke-width:3px

    style PolicyTool fill:#fff,stroke:#333,stroke-width:2px

    style RagTool fill:#fff,stroke:#333,stroke-width:2px

    style ResponsePass fill:#fff,stroke:#333,stroke-width:2px

    

    %% Subgraph and Tool styling

    style Synthesis_Engine fill:#eceff1,stroke:#90a4ae,stroke-dasharray: 5 5

    style Mem0_T fill:#ce93d8,stroke:#4a148c

    style Mem0_B fill:#ce93d8,stroke:#4a148c

    style TAMU_T fill:#ffab91,stroke:#bf360c

    style TAMU_B fill:#ffab91,stroke:#bf360c

    style Atlas fill:#c5e1a5,stroke:#33691e

    style Voyage fill:#80deea,stroke:#006064

    style Langfuse fill:#fff59d,stroke:#fbc02d,stroke-dasharray: 2 2