# Technology and AI Engineering Roadmap

## 1. Project Vision and Learning Objectives

The Multi-Agent Investment System is an AI-powered, agentic application for evidence-grounded investment research and decision support. It will coordinate specialized research and review roles, preserve the evidence behind its outputs, and keep financial calculations and hard risk limits under deterministic control.

The project is also a practical learning path for AI Engineer and GenAI Engineer roles. Implementation should build durable understanding of the architecture, model behavior, evaluation, security, and operations. Generated code is a starting point for review and learning, not a substitute for understanding the design or verifying its behavior.

The system will be built incrementally. Learning exercises will be clearly distinguished from production requirements. This roadmap does not define the four investment strategies.

## 2. Approved Technology Stack

The following technologies and areas are approved for the project and its learning roadmap:

- **Python** for application logic, agent interfaces, deterministic calculations, and tests.
- **Machine learning and deep learning fundamentals** for understanding data, model behavior, training, generalization, and evaluation.
- **LLMs and prompt engineering** for language-based research, synthesis, and structured proposals.
- **Embeddings, vector search, vector databases, and retrieval-augmented generation (RAG)** for locating and grounding outputs in source evidence.
- **LangChain** for model integration, prompts, structured outputs, and tool interfaces.
- **LangGraph** for stateful agent orchestration, conditional routing, retries, and human approval checkpoints.
- **AI agents, tool calling, and multi-agent systems** for bounded research, analysis, and independent review responsibilities.
- **LangSmith** for tracing, debugging, evaluation, and observability of model and agent workflows.
- **FastAPI, SQL, and data engineering** for an application API when needed, and for managing validated evidence, workflow state, decisions, and outcomes.
- **AI security and guardrails, automated testing, MLOps/LLMOps, Docker, CI/CD, and cloud services** for safe, reproducible, monitored operation.

No model provider, SQL product, vector database, cloud provider, or deployment platform is selected by this document. Those choices require a concrete use case, evaluation, and approval under the adoption policy below.

## 3. Complete AI Skills Roadmap

The names and numbering below define the approved 25-skill learning roadmap.

1. Python for AI Engineering
2. Machine Learning Fundamentals
3. Deep Learning
4. LLM Fundamentals
5. Prompt Engineering
6. Embeddings and Vector Search
7. Vector Databases
8. Retrieval-Augmented Generation (RAG)
9. LangChain
10. LangGraph
11. AI Agents and Tool Calling
12. Multi-Agent Systems
13. LLM Fine-Tuning (LoRA/QLoRA)
14. PyTorch
15. Natural Language Processing (NLP)
16. Transformers and Attention Mechanisms
17. FastAPI and Backend Development
18. SQL and Data Engineering for AI
19. LLM Evaluation and Observability
20. AI Guardrails and Security
21. MLOps and LLMOps
22. Docker and CI/CD
23. Cloud AI Services (AWS, Azure, or GCP)
24. Model Serving and Inference Optimization
25. AI Application Testing and Monitoring

## 4. Six Learning Phases

Phases sequence the learning work. A phase may begin with a small learning exercise before its capability is needed in the production path.

### Phase 1 — Foundations

Build Python fluency, clean module boundaries, data validation, deterministic financial calculations, and an automated testing habit. Study machine learning and deep learning fundamentals to understand training, validation, generalization, metrics, and failure modes. Keep financial rules reproducible and independently testable.

### Phase 2 — GenAI

Learn LLM capabilities and limits, prompt design, structured outputs, embeddings, vector search, and RAG. Build small, measurable retrieval and generation exercises around traceable source material. Test whether answers are supported by retrieved evidence before expanding the workflow.

### Phase 3 — Frameworks

Use LangChain for replaceable model, prompt, structured-output, and tool interfaces. Use LangGraph for explicit stateful workflows, conditional routes, retries, and approval checkpoints. Learn bounded tool calling and multi-agent design. Instrument calls and workflows with LangSmith.

### Phase 4 — Application Development

Build the research application around explicit interfaces and evidence records. Introduce SQL persistence for validated evidence and workflow artifacts. Add FastAPI when an application API is required. Make model roles configurable and preserve provider independence.

### Phase 5 — Production AI

Strengthen evaluation, security, operational visibility, and deployment practices. Add regression evaluations, adversarial cases, secure configuration, monitoring, Docker, and CI/CD quality gates. Introduce a selected cloud service only after deployment needs and data-handling constraints are understood and approved.

### Phase 6 — Advanced Topics

Pursue advanced exercises when they have a clear learning objective or the running application demonstrates a measurable need. Topics include PyTorch, transformer internals, fine-tuning, NLP classifiers, model serving, and inference optimization. Compare results against the existing evaluated baseline before considering production use.

## 5. Technology Responsibilities

| Technology or capability | Responsibility and reason it belongs |
|---|---|
| Python | Implements application behavior, clear agent contracts, and deterministic calculations in a testable language suited to AI engineering. |
| LangChain | Provides common interfaces for model calls, prompts, structured outputs, and tools, helping keep providers replaceable. |
| LangGraph | Represents multi-step agent workflows as explicit state, routing, retries, and human approval points. |
| RAG | Grounds research and synthesis in retrieved, traceable evidence so outputs can be checked against sources. |
| Embeddings and vector search | Represent and retrieve semantically relevant evidence; retrieval quality must be evaluated. |
| Vector database | Stores and searches embeddings when data volume, filtering, concurrency, or operational requirements justify a dedicated store. Product selection remains open. |
| LangSmith | Traces model and agent execution and supports debugging and evaluation; traces must follow data-redaction rules. |
| SQL and persistence | Stores structured evidence references, workflow records, decisions, and outcomes for reproducibility and analysis. The SQL product remains open. |
| FastAPI | Exposes application capabilities through an API when a user interface or external client needs one. |
| Testing and evaluation | Checks deterministic correctness, workflow behavior, evidence grounding, model quality, and regressions. |
| Security and guardrails | Protects secrets and data, limits tools, validates outputs, and prevents untrusted content from controlling privileged actions. |
| MLOps/LLMOps, Docker, and CI/CD | Supports reproducible builds, controlled changes, evaluation gates, deployment, and ongoing monitoring. |
| Cloud AI services | May provide hosted models or deployment capabilities after a provider, security needs, and operational tradeoffs are evaluated and approved. |

## 6. Model Routing and Role Switching

Agent roles describe responsibilities, not dedicated model instances. A research, critic, extraction, or synthesis role may use the same model when it meets that role's quality requirements. The system must not assume that each agent needs a separate model.

Model assignments must be configurable by role and task. Provider-specific integrations belong behind replaceable interfaces. Routing decisions should account for:

- Task complexity and required capability.
- Evidence sensitivity and output constraints.
- Quality and safety results from the relevant evaluation set.
- Latency and cost limits.
- Availability and operational reliability.

Use evaluation results to select the least costly, lowest-latency option that meets the task's acceptance criteria. Route to a more capable model, retry, or request human review only through explicit workflow rules. Record the selected model and relevant configuration with traces and outputs so results can be reproduced and compared. Model confidence alone does not establish correctness or investment merit.

## 7. Core Versus Advanced Technologies

### Core for the initial architecture

- Python and testable application interfaces.
- LangChain for model integration, prompts, structured outputs, and tool boundaries.
- LangGraph for controlled stateful workflows and human approval checkpoints.
- RAG and traceable evidence handling.
- Configurable model roles with provider-independent adapters.
- SQL-backed persistence, with the specific product selected later.
- LangSmith tracing and evaluation with appropriate redaction.
- Automated tests, deterministic risk controls, secure configuration, and human review.

FastAPI is introduced when an API is needed. Docker and CI/CD are introduced as deployment and collaboration needs emerge. Embeddings and vector search are part of the retrieval path; a dedicated vector database is introduced only when measured retrieval or operational requirements justify it.

### Advanced or conditional

Deep learning implementation, PyTorch, LLM fine-tuning with LoRA/QLoRA, NLP classifiers, custom model serving, and inference optimization are advanced learning or application capabilities. Cloud-specific services are also conditional. Add these when a learning objective is explicit or measurements show a production need such as quality, latency, throughput, privacy, or cost that the current design cannot meet.

Machine learning and deep learning fundamentals remain part of the learning plan even when custom model training is not part of the initial application. A skill can be learned through a contained exercise without adding its tools to the production dependency set.

## 8. Technology and Dependency Adoption Policy

Before adopting a new framework, dependency, database, provider, or infrastructure service, prepare a short decision proposal that states:

1. The user or engineering problem it solves.
2. Why the current approved stack is insufficient.
3. Viable alternatives, including a simpler or in-house option.
4. Expected learning value and application benefit.
5. Financial cost, operational burden, security and data implications, and maintenance risk.
6. How the choice will be tested and how it could be replaced or removed.

Obtain the owner's approval before adoption. Prefer the smallest dependency set that satisfies the need. Keep learning-only tools out of the production runtime unless separately justified and approved. Record material technology decisions in project documentation.

## 9. Engineering and Safety Standards

- **Testing:** Cover deterministic calculations and risk limits with unit tests. Add interface and workflow tests, failure cases, and end-to-end checks as components are introduced. Tests and human review are required before merging.
- **AI evaluation:** Maintain representative cases for evidence support, citation quality, structured-output validity, routing, safety behavior, latency, and cost. Run evaluations when prompts, models, tools, or workflow logic change.
- **Structured outputs:** Define expected schemas and validate model output before it enters downstream logic. Handle malformed, incomplete, or unsupported responses explicitly.
- **Traceability:** Preserve source references and relevant timestamps, input and configuration versions, model and prompt versions, tool activity, and agent outputs needed to explain and reproduce a proposal.
- **Secure configuration:** Keep credentials outside source control. Apply least privilege, redact sensitive data from logs and traces, validate external input, and treat retrieved documents and tool results as untrusted content.
- **Human approval:** Present decision proposals for human review at explicit workflow checkpoints. Record approval or rejection as a distinct user action.
- **Deterministic financial controls:** Use tested deterministic code for financial calculations and hard risk limits. LLMs may research, interpret, synthesize evidence, and propose actions; they cannot change or override those controls.
- **Failure handling:** Surface missing, stale, conflicting, or invalid evidence. Fail closed when required data or controls are unavailable rather than silently treating uncertainty as reliable output.

## 10. Learn While Building Workflow

Use this workflow for each meaningful feature:

**Understand → Design → Implement with Codex → Review → Test → Document**

- **Understand:** Explain the problem, relevant concepts, constraints, and acceptance criteria.
- **Design:** Compare suitable approaches, document the selected design and tradeoffs, and identify risks and tests.
- **Implement with Codex:** Build a small, reviewable change with clear interfaces and readable code.
- **Review:** Walk through the implementation, assumptions, and generated changes; confirm they match the design.
- **Test:** Run relevant automated tests and AI evaluations; inspect failures and edge cases.
- **Document:** Record usage, behavior, limitations, and material architectural decisions so the learning and reasoning remain available.

## 11. Skill-to-Project Mapping

| # | Skill | Project component or explicit learning exercise |
|---:|---|---|
| 1 | Python for AI Engineering | Implement readable application modules, typed interfaces, configuration, and deterministic calculations. |
| 2 | Machine Learning Fundamentals | Study data splits, leakage, baselines, metrics, and generalization through small evaluation exercises. |
| 3 | Deep Learning | Learn neural-network training and generalization with a contained educational exercise. |
| 4 | LLM Fundamentals | Compare LLM capabilities, context limits, failure modes, and structured generation behavior. |
| 5 | Prompt Engineering | Create versioned prompts and evaluate instruction following, evidence use, and output consistency. |
| 6 | Embeddings and Vector Search | Build and evaluate semantic retrieval over approved research evidence. |
| 7 | Vector Databases | Compare candidate storage approaches and test one only if measured retrieval needs justify it. |
| 8 | Retrieval-Augmented Generation (RAG) | Produce evidence-grounded research with source references and retrieval evaluations. |
| 9 | LangChain | Implement replaceable model, prompt, structured-output, and tool interfaces. |
| 10 | LangGraph | Model workflow state, conditional routing, bounded retries, and human approval checkpoints. |
| 11 | AI Agents and Tool Calling | Implement agents with explicit inputs, outputs, allowed tools, and execution limits. |
| 12 | Multi-Agent Systems | Coordinate specialized research and review roles and inspect disagreement handling. |
| 13 | LLM Fine-Tuning (LoRA/QLoRA) | Run an optional experiment only when a defined learning goal or evaluated quality gap warrants it. |
| 14 | PyTorch | Complete a contained model-building exercise; add to runtime only with a separately justified need. |
| 15 | Natural Language Processing (NLP) | Practice document parsing, normalization, chunking, and text-quality checks for retrieval. |
| 16 | Transformers and Attention Mechanisms | Study transformer architecture and attention through diagrams or a small educational implementation. |
| 17 | FastAPI and Backend Development | Expose research workflows through a validated API when application clients need it. |
| 18 | SQL and Data Engineering for AI | Design persistence for evidence references, workflow state, decisions, and outcome records. |
| 19 | LLM Evaluation and Observability | Trace workflows in LangSmith and maintain evaluations for quality, safety, latency, and cost. |
| 20 | AI Guardrails and Security | Validate inputs and outputs, restrict tools, protect secrets, and test prompt-injection handling. |
| 21 | MLOps and LLMOps | Version prompts, models, configurations, evaluation results, and deployments. |
| 22 | Docker and CI/CD | Create reproducible environments and automated quality gates for changes. |
| 23 | Cloud AI Services (AWS, Azure, or GCP) | Compare a cloud option against requirements and deploy only after provider and service approval. |
| 24 | Model Serving and Inference Optimization | Measure model latency, throughput, and cost before trying serving or optimization techniques. |
| 25 | AI Application Testing and Monitoring | Build unit, interface, workflow, evaluation, security, and runtime monitoring checks. |
