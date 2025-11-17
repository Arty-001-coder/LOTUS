# LOTUS
# LOTUS Debate System 🧠

An advanced AI debate system that uses reinforcement learning, memory graphs, and sophisticated reasoning rules to conduct structured debates between two AI agents (Pro and Con positions).

## Overview

This system implements a multi-agent debate framework where two AI agents argue opposing sides of a question, supervised by a mentor evaluator that assesses argument quality and provides intervention. The system features memory-based learning, dynamic reasoning rules, and a reward system that encourages high-quality argumentation.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LOTUS Debate System                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐         ┌──────────────┐                 │
│  │  Pro Agent   │◄───────►│  Con Agent   │                 │
│  │ (Debator)    │         │  (Debator)   │                 │
│  └───────┬──────┘         └───────┬──────┘                 │
│          │                        │                         │
│          │    ┌──────────────────────┐                     │
│          └───►│  Mentor Evaluator    │◄────┘               │
│               │  (Quality Assessment)│                      │
│               └──────────┬───────────┘                      │
│                          │                                   │
│                          ▼                                   │
│               ┌──────────────────────┐                      │
│               │   Memory Graph       │                      │
│               │  (Debate History)    │                      │
│               └──────────────────────┘                      │
│                          │                                   │
│                          ▼                                   │
│               ┌──────────────────────┐                      │
│               │  Reasoning Rules     │                      │
│               │    (rules.json)      │                      │
│               └──────────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

## File Descriptions

### 1. `core_debator.py` - Main Debate Engine

**Purpose**: The central orchestrator that manages the entire debate process, implements reinforcement learning mechanisms, and coordinates between agents.

**Key Components**:

- **`ReasoningRulesLoader`**: Loads and manages reasoning rules from `rules.json`, including rule costs and complexity levels.

- **`DebateMemoryGraph`**: A graph-based memory system using NetworkX that:
  - Stores debate history as connected nodes
  - Tracks argument quality, rules used, and performance metrics
  - Enables similarity-based retrieval using FAISS vector search
  - Prevents argument repetition through embedding-based detection
  - Analyzes trends in argument quality and rule effectiveness

- **`StrictEnhancedLotusArtRL`**: The main debate orchestrator that:
  - Manages the reinforcement learning system with reward/penalty mechanisms
  - Implements a stamina system (starts at 100, depletes with each turn)
  - Coordinates between Pro/Con agents and the Mentor
  - Tracks performance metrics across multiple frames (ethical, empirical, pragmatic, emotional)
  - Generates internet-augmented context for debates
  - Produces comprehensive debate analysis and synthesis

**Key Features**:
- **Stamina System**: Each agent starts with 100 stamina points that decrease with each argument
- **Reward Banking**: Agents earn reward points based on argument quality (1-10 scale)
- **Multi-Frame Analysis**: Arguments evaluated across ethical, empirical, pragmatic, and emotional dimensions
- **Memory Integration**: Uses past debate history to inform future arguments
- **Echo Prevention**: Detects and prevents repeated arguments using semantic similarity
- **Rule-Based Reasoning**: Dynamically applies reasoning rules based on context

### 2. `mentor.py` - Evaluator & Coach

**Purpose**: Acts as an impartial judge and coach that evaluates argument quality and provides strategic guidance to debaters.

**Key Components**:

- **`EnhancedMentorEvaluator`**: The main evaluation class that:
  - Evaluates arguments across multiple frames/columns (ethical, empirical, pragmatic, emotional)
  - Provides quality scores (0-10) and detailed feedback
  - Integrates internet data via web search for factual grounding
  - Implements "Thousand Brains" architecture with parallel frame processing
  - Provides strategic interventions when agents struggle

**Key Features**:

- **Multi-Frame Evaluation**: 
  - Evaluates each argument from 4 different perspectives
  - Aggregates scores for comprehensive quality assessment
  - Detects which frames are underutilized

- **Internet Context Integration**:
  - Fetches real-time data via Brave Search API
  - Filters and caches relevant information for debate context
  - Provides evidence-based grounding for arguments

- **Mentor Intervention**:
  - Analyzes debate patterns and agent performance
  - Provides strategic suggestions (e.g., "Try a new frame", "Use underexplored rules")
  - Implements coaching mode with step-by-step guidance
  - Introduces "wild card" challenges to prevent stagnation

- **Final Synthesis**:
  - Decomposes the debate question
  - Processes arguments through parallel frames
  - Integrates insights into a unified synthesis
  - Includes opponent modeling and meta-reflection

### 3. `rules.json` - Reasoning Knowledge Base

**Purpose**: A comprehensive catalog of reasoning rules, logical frameworks, and cognitive strategies that guide how agents construct arguments.

**Structure**:
```json
{
  "category_name": {
    "rule_id": {
      "rule": "Description of the reasoning principle",
      "framework": "Guiding questions for applying this rule",
      "keywords": ["trigger", "words"],
      "complexity": "low|medium|high",
      "cost": 0-4
    }
  }
}
```

**Main Categories**:

1. **Foundational Logical Rules**:
   - Causation analysis
   - Evidence hierarchies
   - Counterfactual reasoning
   - Bayesian reasoning
   - Analogical reasoning
   - Generalization boundaries

2. **Awareness Rules**:
   - Complexity awareness (matching reasoning depth to question complexity)
   - Situational awareness (understanding debate context and adapting strategy)
   - Emotional awareness (detecting and regulating emotions in reasoning)
   - Cognitive bias awareness
   - Link awareness (mapping interconnected concepts)

3. **Cognitive Scaffolding Rules**:
   - Argument decomposition
   - Opponent modeling
   - Meta-reasoning and self-critique
   - Synthesis strategies

4. **Domain-Specific Rules**:
   - Scientific reasoning
   - Ethical frameworks
   - Economic analysis
   - Historical reasoning
   - And many more...

**Rule Attributes**:
- **Cost**: Energy/stamina cost to apply the rule (0-4)
- **Complexity**: Sophistication level required (low/medium/high)
- **Keywords**: Trigger words that suggest when to apply the rule
- **Framework**: Practical questions to guide application

## How They Work Together

### Debate Flow

1. **Initialization** (`core_debator.py`):
   ```
   - Load reasoning rules from rules.json
   - Initialize memory graph (empty)
   - Initialize mentor evaluator
   - Set starting stamina (100 for each side)
   - Fetch internet context for the debate topic
   ```

2. **Turn Execution**:
   ```
   For each debate turn:
     ┌─ Pro Agent generates argument
     │  ├─ Selects relevant reasoning rules (from rules.json)
     │  ├─ Accesses memory graph for context
     │  ├─ Uses internet data for grounding
     │  └─ Produces argument
     │
     ├─ Mentor evaluates Pro argument (mentor.py)
     │  ├─ Scores across 4 frames
     │  ├─ Updates memory graph
     │  ├─ Calculates RL rewards/penalties
     │  └─ Updates stamina
     │
     ├─ Con Agent generates counter-argument
     │  ├─ Same process as Pro
     │  └─ Can see Pro's previous argument
     │
     └─ Mentor evaluates Con argument
        └─ Same evaluation process
   ```

3. **Memory Updates** (`core_debator.py`):
   ```
   After each argument:
     - Add argument node to memory graph
     - Create edges to related previous arguments
     - Update rule effectiveness statistics
     - Track frame usage patterns
     - Prevent echo by checking similarity
   ```

4. **Adaptive Intervention** (`mentor.py`):
   ```
   When agent struggles (low quality score):
     - Analyze memory graph patterns
     - Identify underused effective strategies
     - Suggest frame switching
     - Provide coaching guidance
     - Challenge with "wild card" rules
   ```

5. **Termination**:
   ```
   Debate ends when:
     - Either side's stamina drops below threshold (20)
     - Maximum turns reached
     - Arguments become repetitive
   ```

6. **Final Synthesis** (`mentor.py`):
   ```
   - Decompose original question
   - Process through parallel frames
   - Integrate best insights
   - Model strongest counterarguments
   - Provide meta-reflection
   - Generate performance summary
   ```

## Key Innovations

### 1. Reinforcement Learning Integration
- Agents receive immediate feedback through reward/penalty system
- Rewards tied to argument quality (1-10 scale)
- Penalties for poor reasoning or rule misuse
- Cumulative reward tracking influences strategy

### 2. Graph-Based Memory
- Arguments stored as nodes with rich metadata
- Semantic similarity enables contextual retrieval
- Trend analysis reveals strategy effectiveness
- Prevents repetition through echo detection

### 3. Dynamic Rule Application
- 70+ reasoning rules across multiple categories
- Context-aware rule selection based on keywords
- Cost-benefit analysis for rule usage
- Adaptive learning from rule effectiveness

### 4. Multi-Frame Architecture
- "Thousand Brains" inspired parallel processing
- 4 frames: ethical, empirical, pragmatic, emotional
- Independent evaluation then synthesis
- Prevents single-perspective bias

### 5. Internet-Grounded Reasoning
- Real-time web search integration
- Evidence caching for consistent context
- Fact-checking and data validation
- Progressive context updating during debate

## System Requirements

- Python 3.8+
- Ollama with `llama3.2:3b` model
- Required packages:
  - `requests`
  - `numpy`
  - `sentence-transformers`
  - `faiss-cpu` or `faiss-gpu`
  - `networkx`

## Usage

```bash
# Start Ollama server
ollama serve

# Pull required model
ollama pull llama3.2:3b

# Run debate
python core_debator.py
```

The system will prompt you to enter a custom debate question, then conduct a full debate with analysis.

## Output

The system produces:

1. **Console Output**: Real-time debate turns with Pro/Con arguments, stamina, and reward points
2. **`output.txt`**: Complete debate history with all metadata, evaluations, and analysis
3. **`debug_log.txt`**: Detailed logging for debugging and analysis

## Example Debate Turn

```
PRO: The development of AGI within the next decade is unlikely because current 
     neural architectures lack the necessary abstraction capabilities and 
     transfer learning mechanisms observed in biological intelligence.
     [Rules: causation_analysis, evidence_hierarchies]
     Stamina: 85 | Reward Points: 47

CON: However, recent transformer models demonstrate emergent capabilities at 
     scale that weren't predicted, suggesting that architectural limitations 
     may be overcome through computational scaling rather than fundamental 
     redesign.
     [Rules: counterfactual_reasoning, evidence_hierarchies]
     Stamina: 82 | Reward Points: 51
```

## Advanced Features

- **Situational Awareness**: Agents understand debate context and adapt strategy
- **Emotional Intelligence**: Recognition and regulation of emotional content in arguments
- **Cognitive Bias Detection**: Identification and mitigation of reasoning biases
- **Synthesis Levels**: Graduated levels of integration (simple → creative → transcendent)
- **Opponent Modeling**: Strategic prediction of counterarguments
- **Meta-Reasoning**: Self-critique and reflection on reasoning process

## Future Enhancements

- Multi-agent tournaments with evolving strategies
- Real-time human participation as third party
- Integration with additional LLM backends
- Advanced synthesis with cross-debate learning
- Automated rule discovery and refinement

## License

This is a research prototype for exploring multi-agent reasoning systems and debate-based knowledge synthesis.

---

**Built with**: Python, Ollama/LLaMA, FAISS, NetworkX, Sentence Transformers
**Architecture**: Multi-Agent RL with Graph Memory and Dynamic Rule-Based Reasoning
