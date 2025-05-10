Here is the confirmed wiki structure, with the understanding that diagram sections will contain Mermaid syntax:

**Confirmed Repository Wiki Structure (Optimized for Mermaid Generation):**

1. **Home/Overview** (Purpose, Goals, Context \- Synthesized from README/code analysis)  
2. **Repository Snapshot / Key Statistics** (File count, Languages, AWS services, Technologies \- Analyzed from repo contents)  
3. **Setup & Local Development** (Prerequisites, Installation, Configuration, Running \- Extracted from documentation, scripts, config)  
4. **Architecture & System Design**  
   * **Purpose:** Explain the system's overall structure and key design choices.  
   * **Expected Content (from repo):**  
     * High-Level Architectural Description: Synthesized from codebase structure, key components, and comments.  
     * **Architecture Diagram (Mermaid):** Generate a Mermaid graph definition showing major components and their high-level relationships, inferred from code structure and dependencies.  
     * Key Design Principles or Patterns (if discernible from code/comments).  
     * Architecture Decision Records (ADRs) (If ADR files exist, link or include content).  
   * **Guideline:** Synthesize description and generate Mermaid code for the architecture diagram based on inferred components and relationships.  
5. **Codebase Structure & Key Components** (Directory structure, Module breakdown, Entry points \- Inferred/Extracted)  
6. **Building, Testing, and Deployment** (Build process, Testing, Deployment steps \- Extracted)

**Repository Type-Specific Sections (Include based on repo analysis \- Content extracted/inferred, with Mermaid Diagrams):**

7. **Frontend Specifics (If Frontend Repo)**  
   * **Purpose:** Detail frontend-specific aspects.  
   * **Expected Content (from repo):** (as before, plus diagram)  
     * Component Overview, State Management, API Communication details.  
     * **Component Interaction/Flow Diagram (Mermaid):** Generate a Mermaid diagram showing how major frontend components interact or a simplified data flow within the client-side application, if discernible.  
   * **Guideline:** Extract frontend specifics. Generate Mermaid diagram for component interactions or flow if feasible from code.  
8. **Data Processing Specifics (If Data Processing Repo)**  
   * **Purpose:** Detail data sources, processing logic, and data outputs.  
   * **Expected Content (from repo):** (as before, plus diagrams)  
     * Data Sources, Data Model/Schema, Processing Logic, Data Outputs details.  
     * **Data Flow Diagram (Mermaid):** Generate a Mermaid diagram illustrating the flow of data through processing steps or pipelines, inferred from code logic and configuration.  
     * **Sequence Diagram (Mermaid):** For key processing steps or interactions, generate a Mermaid sequence diagram illustrating the order of operations between different parts of the data processing code.  
   * **Guideline:** Extract data processing specifics. Generate Mermaid diagrams for data flow and key sequences if discernible from code logic and structure.  
9. **Infrastructure Specifics (If Infrastructure Repo \- IaC)**  
   * **Purpose:** Document the infrastructure resources and their configuration.  
   * **Expected Content (from repo):** (as before, plus diagram)  
     * Managed Resources, Configuration Details, Dependencies/Relationships details.  
     * **Infrastructure Diagram (Mermaid):** Generate a Mermaid diagram representing the relationships between provisioned infrastructure resources based on the IaC code.  
   * **Guideline:** Extract IaC specifics. Generate a Mermaid diagram of resource relationships as defined in the IaC code.

**Final General Guidelines for LLM Content Generation:**

* **Source Adherence:** Strictly use information present within the repository files.  
* **Synthesize and Analyze:** Read and synthesize information for descriptions. Perform analysis for the "Repository Snapshot."  
* **Generate Mermaid Diagrams:** For sections requiring diagrams, analyze the relevant code, configuration, and structure to infer the relationships, flows, or sequences. **Generate the output directly in Mermaid syntax within a code block.**  
* **Indicate Inference & Gaps:** Explicitly state that the diagrams and descriptions are *inferred* from the codebase and may not capture all nuances or external dependencies not present in the repo. Note if information for any section or diagram was insufficient.  
* **Use Markdown & Mermaid Syntax:** Format the wiki pages using Markdown. Embed generated diagrams using Mermaid code blocks (e.g., mermaid\\n\[Mermaid code here\]\\n).  
* **Objective Tone:** Maintain a neutral, technical tone.  
* **Cross-referencing:** Link between relevant sections using relative wiki links.

