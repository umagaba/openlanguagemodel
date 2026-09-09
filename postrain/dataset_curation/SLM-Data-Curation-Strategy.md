# **Domain-Specific SLM Pretraining: Data Curation & Processing Guide**

---

## **Part A — The Four-Stage Pipeline (applies to every domain)**

\[1\] SOURCE DISCOVERY  →  \[2\] FILTERING  →  \[3\] QUALITY ENHANCEMENT  →  \[4\] EVALUATION  
---

### **Stage 1 — Source Discovery**

**Do**

* Prioritize sources by **authority, relevance, coverage, and freshness**:  
  * **Tier 1 – Organized/authoritative:** government databases, standards bodies, curated repositories. High trust and useful for domain facts, but often organized rather than prose.  
  * **Tier 2 – Community/expert text:** Q\&A forums, technical documentation, and published articles. Higher volume, but quality varies and requires filtering.  
  * **Tier 3 – General web data:** large web corpora such as FineWeb/FineWeb-Edu. Useful mainly for general language fluency rather than deep domain knowledge.  
  *   
* Check whether an existing domain-specific pretraining or instruction dataset is available before collecting data from scratch.  
* Record the source, license, collection date, and version for every dataset. Domain knowledge changes over time, so freshness matters.  
* Decide early whether the domain requires **structured knowledge or a neurosymbolic component**. If it does, collect structured entities, relations, rules, and provenance alongside raw text.

**Don't**

* Don't assume that more data means better data. A small amount of authoritative domain data can be more valuable than a large noisy corpus.  
* Don't collect data without checking its licensing and permitted use.  
* Don't treat all sources equally. Weight sources according to their authority and reliability.  
* 

### **Stage 2 — Filtering**

**Do**

* Apply filtering in layers, cheapest first:  
  * Rule-based: language ID, length bounds, boilerplate/HTML stripping, exact near-duplicate removal (MinHash-LSH is the standard technique for large corpora).  
  * Heuristic quality: symbol-to-word ratio, repetition ratio, stopword presence — the same class of filters.  
  * Classifier-based: train (or reuse) a lightweight scorer for domain relevance and quality — the FineWeb-Edu pattern of using an LLM to annotate a sample (\~500K docs), training a small regression/classifier on those annotations, then applying a tuned threshold to the full corpus, is directly reusable for any domain (e.g., "is this a genuine agricultural extension answer?" instead of "is this educational?").  
* Filter per-source-type separately. A filter tuned for forum Q\&A will misjudge government reports and vice versa.

**Don't**

* Don't use a single quality filter across structurally different sources (tabular nutrition databases vs. mental-health forum posts, all need different rules).  
* Don't filter only on "quality" and forget domain relevance — a beautifully written article that isn't actually about your domain still needs to be dropped.

  ### **Stage 3 — Quality Enhancement**

**Do**

* Deduplicate across the entire corpus, not just within individual sources.  
* Preserve useful domain-specific language such as jargon, abbreviations, code-switching, and regional terminology.  
* Rewrite relevant but poorly written text when the domain has limited high-quality data instead of discarding it automatically.  
* Convert structured or tabular information into natural-language training examples when the target model needs to generate text from that information.  
* Keep structured data in its original form as well when it may later be used for retrieval, knowledge graphs, or neurosymbolic reasoning.  
* Decontaminate the training corpus against your evaluation datasets and public benchmarks.

**Don't**

* Don't over-clean the data and remove useful domain-specific language.  
* Don't use benchmark documents as training data when they may overlap with your evaluation set.  
* Don't discard structured information simply because the model will ultimately be trained on text.  
* 

### **Stage 4 — Evaluation**

**Do**

* Use a layered eval suite, because no single metric is sufficient:  
  * **Perplexity** — cheap, useful for tracking training progress and comparing checkpoints of the same architecture, but it mostly reflects general fluency, not domain knowledge, so don't rely on it alone.  
  * **Downstream task benchmarks** — MCQ-style (domain QA, classification) and generation-style (summarization, extraction) tasks specific to the domain. Build or reuse a domain benchmark (see table below) rather than only general ones like MMLU.  
  * **Reference-based metrics** (BLEU/ROUGE/BERTScore) — useful when you have gold answers, but weak for open-ended generation; pair with something else.  
  * **LLM-as-judge** — fast and scalable for multidimensional quality (helpfulness, factuality, domain-appropriateness), but pair it with periodic human spot-checks since judge models have known bias and consistency issues, especially when judging models similar to themselves.  
  * **Human evaluation** — the highest-signal but most expensive tier; reserve it for final checkpoints and for domains with real-world stakes (mental health, finance, agriculture advisory).  
* Build a small, held-out, in-domain benchmark early, even 100–500 expert-reviewed questions, rather than waiting until the end. AgReason (100 expert questions) and AgriBench-13K (13 task types across 9 sub-domains) both show that even modest expert-curated sets meaningfully separate model quality.  
* For sensitive domains (mental health, medical, finance), add a dedicated safety/harm eval pass on top of accuracy metrics — a technically "correct-sounding" wrong answer is a different kind of failure than a low BLEU score.

**Don't**

* Don't report perplexity as your only evidence a domain SLM "works" — it doesn't track domain accuracy well and can keep improving even after MCQ-style performance has saturated.  
* Don't use a general-purpose benchmark (MMLU, HellaSwag) as your primary success metric for a narrow-domain SLM — it will systematically under-credit domain specialization.  
* Don't evaluate only with the same LLM family you used to generate/filter your training data — this risks circular bias (the judge rewarding style choices it "prefers" from its own family, not real quality).

---

## **Part B — Knowledge Graphs and Neurosymbolic AI**

Knowledge graphs provide structured representations of domain knowledge by connecting entities through explicit relationships. They can complement a domain SLM by providing context, enabling structured reasoning, supporting explainability, and enforcing domain constraints.

A KG should not be treated as a static dataset. It is a lifecycle involving **design, ingestion, extraction, enrichment, storage, consumption, evaluation, and maintenance**. [EMPWR](https://wiki.aiisc.ai/index.php?title=EMPWR:_Knowledge_Graph_Development_Platform) follows this kind of lifecycle and combines symbolic and data-driven methods to construct and maintain KGs from structured, semi-structured, and unstructured sources.

### **Stage 1 — Decide and Design**

**Do**

* Decide early during Stage 1 Source Discovery whether the domain needs structured knowledge. This decision affects what data you collect: entities, relationships, rules, attributes, and provenance, not just prose.  
* Define the domain entities and relationships that the KG needs to represent.  
* Reuse and extend existing authoritative KGs or structured sources where possible rather than starting from zero.  
* Support both:  
  * **Top-down construction:** enrich an existing KG with domain-specific knowledge.  
  * **Bottom-up construction:** build a KG from entities and relations extracted from domain data.  
* Design for provenance, temporal information, domain specificity, and modularity from the beginning.

**Don't**

* Don't build a KG without first identifying what structured relationships or constraints the domain actually requires.  
* Don't assume that a general-purpose KG will adequately represent a specialized domain.

  ### **Stage 2 — Knowledge Ingestion and Extraction**

**Do**

* Ingest knowledge from structured, semi-structured, and unstructured sources.  
* Use neural methods such as NLP models or LLMs for entity and relation extraction from text.  
* Use deterministic or rule-based methods where the source already provides reliable structure.  
* Keep the original source associated with extracted facts so that every relation can be traced back to its evidence.  
* Separate extracted facts from inferred facts. A relation directly supported by a source should not be treated the same as one inferred by a model.

**Don't**

* Don't treat automatically extracted relations as ground truth.  
* Don't discard the source context behind a triple.

  ### **Stage 3 — Enrichment and Curation**

**Do**

* Link extracted entities to authoritative external sources where possible.  
* Resolve aliases and duplicate entities before merging information.  
* Add metadata such as source, provenance, confidence, timestamp, and extraction method.  
* Check for conflicting relations across sources.  
* Use domain experts to validate sampled facts in high-stakes domains.  
* Keep the KG updateable as the underlying domain changes.

**Don't**

* Don't merge multiple sources without resolving differences in schemas, entities, and relation definitions.  
* Don't treat enrichment as a one-time step. New sources and updated facts should be incorporated throughout the KG lifecycle.

  ### **Stage 4 — Neurosymbolic Integration**

**Do**

* Combine the strengths of neural and symbolic methods:  
  * **Neural models** handle language understanding, extraction, generalization, and interaction with unstructured data.  
  * **Symbolic structures** represent explicit facts, relationships, rules, and constraints.  
* Use the KG to provide:  
  * **Contextualization:** connect a query to relevant domain entities and relationships.  
  * **Reasoning:** traverse explicit relationships and apply domain rules.  
  * **Explainability:** expose the facts or graph paths supporting an answer.  
  * **Constraint checking:** verify that generated outputs satisfy known domain rules.  
* Choose the integration pattern based on the application:  
  * **Graph-RAG:** retrieve relevant graph information at inference time.  
  * **KG-to-text training:** convert structured knowledge into training examples.  
  * **Rule-based validation:** check model outputs against explicit constraints.  
  * **Hybrid reasoning:** combine neural predictions with symbolic rules or graph traversal.  
* Track which facts, relations, and rules contributed to an answer so that the system can be audited.

**Don't**

* Don't expect the KG to replace the neural model. The two components serve different purposes.  
* Don't use symbolic constraints as an afterthought. The need for structured knowledge should influence data collection and KG design from the beginning.  
* Don't assume that a reliable KG automatically produces a reliable neurosymbolic system. Extraction, retrieval, reasoning, and integration can each introduce errors.

  ### **Stage 5 — Evaluation and Maintenance**

**Do**

Evaluate both the KG and the integrated system.

**KG-level evaluation**

* **Coverage:** how much of the relevant domain is represented.  
* **Precision:** how many sampled facts are correct.  
* **Consistency:** whether facts and relations contradict each other.  
* **Provenance:** whether facts can be traced back to their sources.  
* **Freshness:** whether facts remain current as the domain changes.

**System-level evaluation**

* **Constraint-violation rate:** how often the system produces outputs that violate known rules.  
* **Reasoning accuracy:** whether the system reaches the correct conclusion from the relevant graph facts.  
* **Reasoning faithfulness:** whether the graph path or symbolic evidence actually supports the answer.  
* **Explainability:** whether the system can identify the knowledge used to produce an answer.

**Don't**

* Don't evaluate only the final answer.  
* Don't assume a large KG is necessarily a useful KG.  
* Don't treat KG maintenance as optional. Domain knowledge can change, and stale facts can lead to incorrect model outputs.

---

## **Part C — Domain-Specific Data Strategies**

### **1\. Food & Nutrition**

| Stage | What to do |
| ----- | ----- |
| **Sources** | Use authoritative Indian nutrition sources such as IFCT/ICMR-NIN and INDB for structured nutrient information, supplemented by recipe and food-text datasets for natural language coverage. |
| **Filtering** | Prioritize authoritative sources, remove incomplete records, and canonicalize food and recipe names such as "chana masala" and "chole." |
| **Quality enhancement** | Convert structured nutrient records into natural-language training examples while retaining the original structured data for retrieval or KG construction. Preserve regional-language recipe text alongside translations. |
| **Evaluation** | Evaluate nutrient estimation and food-related QA using India-specific meals and serving conventions. Validate a sample with nutrition experts. |

### **2\. Mental Health**

| Stage | What to do |
| ----- | ----- |
| **Sources** | Use appropriately licensed clinical datasets, de-identified support/forum data where permitted, and peer-reviewed clinical literature. |
| **Filtering** | Treat privacy and safety as core filtering requirements. Remove personally identifying information and handle crisis-related content conservatively. |
| **Quality enhancement** | Prioritize clinically grounded terminology and review the corpus for demographic, cultural, and linguistic bias. |
| **Evaluation** | Evaluate both task accuracy and safety. Include harm-related evaluation and expert review before deployment. |

### **3\. Agriculture**

| Stage | What to do |
| ----- | ----- |
| **Sources** | Use government agricultural resources, crop and pest databases, extension material, and agronomy literature. Add imagery or remote-sensing data for multimodal applications. |
| **Filtering** | Filter for crop, region, climate, and season. Advice that is valid in one region may not apply elsewhere. |
| **Quality enhancement** | Preserve structured crop, disease, treatment, and regional information where possible so it can support retrieval or KG-based reasoning. |
| **Evaluation** | Use task-specific evaluation across crop identification, disease, advisory, retrieval, and reasoning. Include expert-reviewed questions and test against current information. |

### **4\. Finance**

| Stage | What to do |
| ----- | ----- |
| **Sources** | Use financial filings, earnings calls, financial research, and appropriately licensed financial news. |
| **Filtering** | Deduplicate heavily syndicated content and preserve document dates and company information. |
| **Quality enhancement** | Preserve numerical values, tables, relationships, and document provenance rather than converting everything into plain text. |
| **Evaluation** | Focus on numerical accuracy, document-grounded QA, multi-step reasoning, and factuality. Test whether answers remain valid as financial information changes over time. |

### **5\. Online Support / Generic Technical (recap — see companion guide for full detail)**

| Stage | What to do |
| ----- | ----- |
| Public sources | Stack Overflow archives, GitHub/GitLab issue trackers, product support forums (e.g., Firefox SUMO), plus FineWeb-Edu for general coherence. |
| Filtering | Filter GitHub Issues down to product-relevant, low-code/high-text content; filter Stack Overflow by vote count and answer-acceptance as a quality proxy. |
| Quality enhancement | Explicit A/B mixture ablation (general-vs-technical ratio) at small scale before committing to a ratio at larger scale. |
| Evaluation | Task success on realistic support tickets/queries, ideally judged against the actual resolution rather than surface fluency; staged evaluation at each scale checkpoint (135M → 350M → 1B) to catch regressions early. |

---

### **Part C.1 — Sources & Licensing**

**Do**

* Create a data manifest before training. For each source, record its license, collection date, source URL, and whether the license permits modification, redistribution, and model training.  
* Prefer sources with clear terms of use and explicit permission for research or data reuse.  
* Check the specific terms of government and institutional datasets rather than assuming that publicly available data is unrestricted.  
* Treat scraped websites, forums, and community-generated content cautiously. Availability online does not automatically mean that the content can be redistributed or used for model training. When scraping is necessary, use a responsible, configurable scraping framework such as [Scrapling](https://github.com/D4Vinci/Scrapling), and respect the target website’s terms of service, robots.txt, rate limits, and applicable copyright/licensing requirements.  
* Keep the original source and license information attached to processed datasets so that provenance is not lost during cleaning and transformation.

**Don't**

* Don't assume that a dataset is reusable simply because it appears in a published paper, GitHub repository, Kaggle, or Hugging Face.  
* Don't assume that public access means unrestricted use.  
* Don't remove licensing and provenance information when creating derived datasets.  
* Don't mix sources with incompatible licenses without checking whether the resulting dataset can legally be redistributed.  
* 

### **Part C.2 — Post-Training: Synthetic Data Curation for SFT**

Pretraining teaches the model to model language and domain content; supervised fine-tuning (SFT) teaches it to follow instructions and behave like an assistant in your domain. This stage runs on a much smaller, much higher-quality dataset — and today it's mostly synthetic (model-generated), not manually created.

When generating synthetic SFT data, **the choice of the data-generation model is critical**. The model used to generate the data should have sufficient **domain knowledge** and should reliably support all the **target languages and language combinations** required by the SFT dataset. A model that lacks knowledge of the target domain or performs poorly in a target language can introduce incorrect, biased, or low-quality examples that are then learned by the student model.

Therefore, before generating the full dataset, **compare 2–3 candidate models** on a small representative set of prompts. Models can be accessed through APIs such as [**Groq**](https://groq.com) or [**OpenRouter**](https://openrouter.ai) using API keys. Evaluate them on criteria such as:

* **Domain knowledge** — Does the model understand the target subject/domain accurately?  
* **Language coverage** — Can it generate high-quality examples in all target languages, including code-mixed or multilingual inputs where required?  
* **Instruction following** — Does it produce the requested task format reliably?  
* **Factuality** — Does it avoid hallucinating domain-specific information?  
* **Consistency** — Does it maintain quality across different prompt types?  
* **Synthetic-data quality** — Are the generated responses diverse, useful, and suitable for SFT?

Based on this comparison, select the best-performing model as the **synthetic-data generator** and use it to generate the larger SFT dataset. Ideally, retain a small human-reviewed evaluation set to periodically verify the quality of the generated data.

**Core generation paradigms**

* **Self-Instruct**: start from a small human-written seed set (the original paper used \~175 seed tasks), then prompt a capable LLM to generate new instructions and responses in the same style, iteratively expanding the set. This was the method behind Alpaca (GPT-3.5-generated data used to fine-tune LLaMA into a 52K-example instruction set from just 175 seeds).  
* **Distillation**: use a strong "teacher" model to generate instruction-response pairs (often including reasoning traces/chain-of-thought, not just final answers) that a smaller "student" model — your domain SLM is fine-tuned on. This is now the dominant approach for SFT data specifically, since teacher-model completions frequently exceed what human writers can produce at the same scale and cost.  
* **Evol-Instruct / self-chat variants**: iteratively increase the complexity or diversity of seed instructions (used by WizardLM/WizardCoder), or have a model converse with itself starting from real seed questions (e.g., Stack Overflow/community questions as seeds) to generate multi-turn dialogue data — directly relevant if your domain SFT set should include multi-turn diagnostic/support conversations.

**Quality pipeline for synthetic SFT data**

Do run synthetic data through a layered filter before training, in this order:

1. Length filter — drop degenerate too-short or runaway too-long generations.  
2. Deduplication — MinHash or embedding-similarity dedup; synthetic generation pipelines often produce near-duplicate instructions when prompted repeatedly from similar seeds.  
3. Quality/faithfulness scoring — use an LLM judge (ideally a different model family than your generator, to avoid circular self-preference) to check the response actually answers the instruction and doesn't hallucinate domain facts.  
4. Toxicity/safety filter — non-negotiable for any domain, critical for mental health and finance.  
5. Diversity sampling — ensure task-type and topic coverage rather than keeping only the highest-scored-but-redundant examples; a dataset that's technically high-quality but narrow will produce a narrow model.

**Do**

* Prioritize quality and diversity over raw dataset size.  
* Keep a separate human-verified evaluation set that never enters the synthetic generation pipeline.

**Don't**

* Don't skip the faithfulness/factuality check on distilled data — teacher models hallucinate confidently, and that confidence transfers to the student model during SFT.  
* Don't rely on a single quality-scoring pass — use multiple signals (length/relevance/factuality/diversity) rather than one LLM-judge score, since any single automated signal has blind spots.  
* Don't use synthetic preference data without a clear and validated definition of what makes one response better than another.

---

## **Part D — Cross-Domain Evaluation Toolkit**

| Metric / Method | Best for | Watch out for |
| ----- | ----- | ----- |
| **Perplexity** | Tracking training progress and comparing checkpoints of the same model | Does not reliably measure domain knowledge or task performance |
| **Downstream MCQ** | Fast, inexpensive comparison across models | Can be sensitive to option ordering and formatting; may saturate early |
| **Generation metrics** (BLEU, ROUGE, BERTScore) | Tasks with reference answers such as summarization and extraction | Weak for open-ended generation and factuality |
| **LLM-as-judge** | Scalable evaluation of multidimensional response quality | Can introduce evaluator bias; calibrate periodically with human evaluation |
| **Human evaluation** | Final validation and safety-critical assessment | Expensive and slow; requires clear criteria and inter-rater agreement |
| **Domain-specific benchmark** | Measuring actual domain specialization | Must reflect the domain's real tasks and failure modes |
| **Contamination check** | Ensuring evaluation data has not leaked into training | Should be performed before reporting benchmark results |
| **KG precision / coverage** | Evaluating a KG before using it for retrieval, reasoning, or constraints | Performance may be much worse for rare entities and relations |
| **Constraint-violation rate** | Testing whether a neurosymbolic system follows symbolic rules | A low rate may simply mean that constraints are rarely triggered; measure trigger frequency as well |
| **Reasoning faithfulness** | Checking whether the facts, relations, or graph paths used by the system actually support the final answer | | A correct answer does not necessarily mean the intended KG or symbolic reasoning path was used |

---

## **Further Explore**

* Penedo et al., *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale* (arXiv:2406.17557) — classifier-based filtering methodology, reused across domain tables above.  
* Maini et al., *Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling* — LLM-based rewriting of noisy text as a quality-enhancement technique.  
* Hua et al., *NutriBench: A Dataset for Evaluating LLMs on Nutrition Estimation from Meal Descriptions*(arXiv:2407.12843) — food/nutrition source-to-benchmark pipeline.  
* Wu et al., *BloombergGPT: A Large Language Model for Finance* (arXiv:2303.17564) — general:domain corpus ratio precedent.  
* AgriGPT: *A Large Language Model Ecosystem for Agriculture* (arXiv:2508.08632) — multi-agent data engine and AgriBench-13K task-typed evaluation.  
* AgReason / AgThoughts (baskargroup.github.io/Ag\_reasoning) — small expert-curated reasoning benchmark example.  
* *Constructing Domain-Specific Evaluation Sets for LLM-as-a-Judge* (arXiv:2408.08808) — methodology for building separability-tested domain eval sets.  
* *From Raw Corpora to Domain Benchmarks: Automated Evaluation of LLM Domain Expertise* (arXiv:2506.07658) — on perplexity's limits and automated domain-benchmark generation.  
* Hoffmann et al., *Training Compute-Optimal Large Language Models* ("Chinchilla") — the \~20:1 token-to-parameter scaling guideline used for token budgeting.  
* *Development of an Indian Food Composition Database* (PMC, 2024\) — ICMR-NIN IFCT 2017 and the Indian Nutrient Databank (INDB) construction methodology.  
* Khanna, Chattopadhyay & Kundu, *INDoRI: Indian Dataset of Recipes and Ingredients and its Ingredient Network*(arXiv:2309.10403).  
* Wang et al., *Self-Instruct: Aligning Language Models with Self-Generated Instructions* — the foundational Self-Instruct/Alpaca-style synthetic SFT data pipeline.  
* Zhou et al., *LIMA: Less Is More for Alignment* — the curated-quality-over-volume finding for SFT data (\~1,000 examples matching 52,000 uncurated).  
* Lambert, *RLHF and Post-Training Book* (rlhfbook.com) — overview of synthetic data's role across SFT, preference/RLHF, and evaluation stages.  
* Lehmann et al., *DBpedia: A Large-scale, Multilingual Knowledge Base Extracted from Wikipedia* — semi-automatic KG construction from semi-structured infoboxes.  
* Suchanek, Kasneci & Weikum, *YAGO: A Core of Semantic Knowledge* — KG built from Wikipedia \+ WordNet alignment.  
* Carlson et al., *Toward an Architecture for Never-Ending Language Learning* (NELL) — continuous automatic extraction from unstructured text.  
* Dong et al., *Knowledge Vault: A Web-Scale Approach to Probabilistic Knowledge Fusion* — Google's microdata/markup-based, confidence-scored KG construction.  
* Vrandečić & Krötzsch, *Wikidata: A Free Collaborative Knowledgebase* — manually/collaboratively curated KG.  
* Hitzler et al., *Neuro-Symbolic Approaches in Artificial Intelligence* — overview of combining symbolic knowledge structures with neural models for constrained reasoning and explainability.

