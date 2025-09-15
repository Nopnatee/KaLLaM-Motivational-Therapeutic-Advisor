“The primary objective of this project is to demonstrate the effectiveness of SEALION models in handling psychological context and dialogue—both by generating meaningful therapeutic sessions with users and by evaluating conversations through automated SEALION-based MISC annotation as an AI coder.”

# Project Proposal: KaLLaM Motivational Therapeutic Advisor

## 1. Introduction

The increasing availability of large language models (LLMs) has opened opportunities for building mental health chatbots that simulate therapeutic dialogue. Yet, most systems are evaluated with surface-level metrics (BLEU, ROUGE, perplexity) that miss psychological nuance, empathy, and safety concerns. And there is no model tune especially for Thai Motivational Therapeutic Advisor. This project tackles the gap by exploring **SEALION models**—a multilingual family of LLMs tailored for Southeast Asia—as engines for both generating meaningful therapeutic conversations and serving as automated evaluators using **Motivational Interviewing Skill Code (MISC) 2.5**.

The broader motivation is to establish an AI advisor that is not only conversationally fluent but also clinically aligned, safe, and capable of evaluating itself and others with interpretable metrics.

## 2. Objectives

* Demonstrate the effectiveness of SEALION models in handling psychological context in therapeutic dialogues.
* Generate structured, effective therapeutic sessions that reflect Motivational Interviewing (MI) principles.
* Evaluate conversations automatically using SEALION-based MISC coding to approximate expert human annotations.
* Benchmark against recent evaluation frameworks (psychological metrics, safety metrics, agentic evaluators).

## 3. Methodology

### 3.1 Datasets

* **EmpatheticDialogues** (Rashkin et al., 2019) as a baseline for empathy-grounded training.
* **Annotated MI datasets** (e.g., MI-TAGS, Pérez-Rosas et al. 2016, or self-annotated corpora with MISC codes).
* Synthetic **benchmark scenarios** (e.g., 100-item safety set from Xu et al., 2024).

### 3.2 Models and Pipelines

1. **Therapeutic Dialogue Generation**

   * Fine-tune or prompt SEALION models for role-play therapy.
   * Align outputs with MI spirit (collaboration, evocation, autonomy) and OARS skills (Open questions, Affirmations, Reflections, Summaries).

2. **Automated Evaluation (AI Coder)**

   * Implement SEALION-based pipelines for MISC 2.5 annotation.
   * Integrate psychological evaluation metrics (empathy, emotion matching, linguistic style matching).
   * Explore **agentic evaluator pipelines** that query external resources to validate safety-critical outputs.

3. **Validation**

   * Compare SEALION-MISC coding with human annotators.
   * Cross-check evaluation quality with psychological metrics (Jiang et al., 2023) and safety metrics (Xu et al., 2024).

### 3.3 Tools & Implementation

* Python pipeline for dialogue generation and annotation.
* Integration with Hugging Face datasets and SEA-Lion API.
* Evaluation dashboards to visualize metrics over sessions.

## 4. Expected Outcomes

* A demonstrator chatbot capable of generating MI-style conversations.
* An automated coding pipeline that approximates human-level MISC annotation.
* Comparative insights: SEALION vs other LLMs (GPT-4, Claude, etc.) in psychological robustness and evaluation accuracy.
* Contribution of **self-scoring AI therapists**, which may reduce annotation costs and accelerate clinical AI research.

## 5. Timeline

* **Month 1–2:** Literature review, dataset preparation, baseline scripts.
* **Month 3–4:** Implement SEALION-based dialogue generation.
* **Month 5–6:** Develop SEALION-MISC annotation coder.
* **Month 7–8:** Integrate psychological & safety metrics, agentic evaluators.
* **Month 9:** Human-in-the-loop validation with experts.
* **Month 10:** Final analysis, reporting, and demonstration.

## 6. Impact

* **Research contribution:** Bridges LLM dialogue generation with interpretable, clinically relevant evaluation.
* **Practical benefit:** Offers a scalable tool to simulate therapeutic sessions and provide real-time feedback on quality.
* **Broader value:** Advances safe, empathetic, and trustworthy AI for mental health contexts in Southeast Asia and beyond.

## 7. References
You can refer to the Citation.md file in the repositary

---