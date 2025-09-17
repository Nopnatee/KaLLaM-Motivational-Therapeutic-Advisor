# Project Proposal: KaLLaM Motivational Therapeutic Advisor

## 1. Introduction

Large language models (LLMs) are transforming how we imagine digital healthcare. They make it possible to design mental health chatbots that not only hold conversations but also guide users through therapeutic dialogue. Yet most current systems remain limited—measured by surface scores like BLEU, ROUGE, and perplexity—while overlooking what truly matters: empathy, psychological nuance, and safety.

In Thailand and across much of Southeast Asia, this gap is even wider. There is no dedicated large language model designed for motivational therapy, and no Thai-specific MISC-annotated resources to support rigorous evaluation. In practice, this means there is currently no model or dataset that fits the requirements for Thai motivational interviewing—leaving millions without culturally aligned AI support that reflects their language, values, and clinical practices.

Our project seeks to close this gap. We propose the “Silver Standard Score”, an evaluation baseline grounded in the Motivational Interviewing Skill Code (MISC) 2.5 [@houck2010misc25]. This framework enables culturally aligned, interpretable assessment of Thai therapeutic dialogue where no such benchmark currently exists. By combining this evaluation standard with SEA-L models trained for Southeast Asia, the project lays the groundwork for the first AI advisor in Thailand that is both clinically credible and locally relevant.

To realize this vision, we will harness SEALION models—a multilingual family of LLMs trained with Southeast Asian data—as both empathetic conversational partners and rigorous evaluators. By adapting these models to Thai therapeutic contexts, we aim to deliver the first system capable of generating and assessing motivational dialogue with cultural and clinical fidelity. The goal is simple but transformative: an AI advisor that is fluent in Thai, safe in practice, and clinically aligned—setting a new standard for mental health technology in Thailand and offering a model for the wider region.

## 2. Objectives

* Demonstrate the effectiveness of SEALION models in handling psychological context in both medical and therapeutic dialogues.
* Have the capability of generating structured, effective therapeutic sessions that reflect Motivational Interviewing (MI) principles.
* Evaluate conversations automatically using SEALION-based MISC coding to approximate expert human annotations as Silver Standard Score.
* Benchmark each MI sessions handled by human, orchestrated LLMs and LLM.

## 3. Methodology

### 3.1 Datasets

* **EmpatheticDialogues** [@rashkin2019towards] as a baseline for empathy-grounded technique in conversations.
* **MI datasets** This project leverages the *EmpatheticDialogues* dataset [@rashkin2019towards] as human based conversations for benchmark.
* Synthetic **benchmark scenarios** using our Orchestrated LLMs and single fragship LLM(s) (e.g., 100-item safety set [@xu2024building]).

### 3.2 Main Models and Pipelines

1. **Therapeutic Dialogue Generation**

   * Prompt engineered SEALION models conversation with real human approved role/real-play for using in evaluation.
   * Align chatbot response with orchestration medical and therapeutic techniques with additional agents.

2. **Automated Evaluation (AI Coder)**

   * Implement SEALION-based pipelines for strict MISC 2.5 [@houck2010misc25] annotation.
   * Integrate psychological evaluation metrics (empathy, emotion matching, linguistic style matching) using certif [@jiang2023psychological].
   * Explore **agentic evaluator pipelines** that query external resources to validate safety-critical outputs.
   * We also incorporate safety-focused and clinically responsible when responding in mental-health contexts [@xu2024building].

3. **Validation**

   * Benchmark models and human dataset with SEALION-MISC coding as Silver Standard Score.
   * Cross-check evaluation quality with psychological metrics [@jiang2023psychological] and safety metrics [@xu2024building].

### 3.3 Tools & Implementation

To build KaLLaM, we rely on a combination of model, data, orchestration, and evaluation infrastructure:

* **Core Models**
  * **SEA-LION** LLMs (AI Singapore, 2024) as the backbone for generation and automated coding.
  * Comparative baselines with GPT-4, Claude, or Gemini for evaluation alignment and benchmark.

* **Datasets**
  * Human *EmpatheticDialogues* [@rashkin2019towards] for benchmark.
  * MI-specific corpora (MISC 2.5–coded transcripts, MI-TAGS, [@houck2010misc25]) for therapeutic skill alignment.
  * Some of synthetic safety benchmark scenarios [@xu2024building] by human role-play for stress-testing in crisis contexts.

* **Conversation Orchestration**
  * Modular **ChatbotManager** and **Orchestrator** components to manage sessions, multi-agent routing (DoctorAgent, PsychologistAgent, SupervisorAgent), and logging.
  * SQLite-based **session/message stores** for simple reproducible conversation histories.

* **Evaluation Infrastructure**
  * SEALION-based MISC annotator trained on local Thai conversational data, ensuring reliable automatic coding of therapist–client and standardize in both English and Thai cultural context for both Thai and English datasets.
  * Psychological evaluation metrics (empathy, emotion matching, linguistic style matching [@jiang2023psychological]) computed on Thai text with higher fidelity than generic LLM metrics for standardized MISC coded datasets.
  * Safety evaluators (SEALION’s agentic evaluator and guideline-based scoring [@xu2024building]), adapted for Thai contexts, able to check guideline adherence and risk management with culturally aligned benchmarks.

* **Visualization & Reporting**
  * Custom dashboards to plot R/Q ratio, % open questions, % complex reflections, and MI-consistent coding over sessions.
  * Comparative charts for empathy and safety scores across models.

## 4. Expected Outcomes

* A demonstrator chatbot capable of generating MI-style conversations.
* An automated coding pipeline that approximates human-level MISC annotation.
* Comparative insights: SEALION vs other LLMs (GPT-4, Claude, etc.) in psychological robustness and evaluation accuracy.
* Contribution of **self-scoring AI therapists**, which may reduce annotation costs and accelerate clinical AI research.

## 5. Potential Impact

* **Research contribution:** Advances the link between LLM dialogue generation and clinically interpretable evaluation, incorporating psychological and safety metrics that go beyond surface fluency. With SEALION’s locally trained models, the project enables evaluation in Thai and regional languages that mainstream benchmarks often neglect.
* **Practical benefit:** Delivers a scalable platform to simulate therapeutic sessions and provide automated, real-time feedback, powered by SEALION’s annotators that capture culturally specific conversational nuances.
* **Broader value:** Demonstrates how regionally trained models like SEALION can support safe, empathetic, and trustworthy AI for mental-health contexts in Southeast Asia and contribute to setting global standards for responsible healthcare AI.

## 6. Disclaimer

* **Not a substitute for professional care:** KaLLaM is a research and evaluation framework. It is not designed to replace licensed therapists, psychologists, or medical professionals.
* **Intended use:** Outputs are meant for research, training, and system evaluation. They should not be treated as definitive diagnoses, clinical advice, or emergency guidance.
* **Safety first:** Users in crisis or experiencing severe distress should immediately seek help from qualified health professionals or local emergency services even if the model faild to reccomended so.
* **Model limitations:** The project does not involve fine-tuning, all responses are generated through prompt engineering with existing LLMs. This means results can be inconsistent and should be expected to improve significantly with future fine-tuning or domain-specific training.

## 7. References
You can refer to the Citation.md file in the repositary

---