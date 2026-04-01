# Review Comments for EAAI-25-21508

**Manuscript:** "A Reinforcement Learning-Based Spatiotemporal Dynamic Multi-Graph Ensemble Framework for Multi-Station Air Quality Prediction"

**Authors:** Distinguished Professor Yan Ke, Xiang Ma; Yufei Qian; Jing Huang; Yang Du

**Journal:** Engineering Applications of Artificial Intelligence

**Editor:** Patrick Siarry, Editor In Chief

**Decision:** Major Revision

**Revision Due Date:** Apr 16, 2026

**Editor Comments:**

While the reviewers found the paper interesting, they are all agreed that very extensive revisions will be necessary if the paper is to be acceptable for publication in Engineering Applications of Artificial Intelligence, to make the paper of interest and benefit to a wider cross-section of our readers.

Please indicate, when submitting a revised version, what changes have been made to the paper. Also, the revision of the paper should have no acronyms in the title and keywords, and all acronyms in the abstract must be defined.

---

## Reviewer #1

designed to address spatiotemporal heterogeneity in air quality forecasting. The architecture integrates a Gated Temporal Convolutional Network, a Hybrid Graph Learning Module , and an Adaptive Spatiotemporal Attention Mechanism. The final prediction is dynamically fused using a Deep Deterministic Policy Gradient agent.
The primary concern regarding the validity of this work is the choice of the dataset and the forecasting horizon:
The study utilizes a Beijing air quality dataset spanning March 2013 to February 2017. Given the rapid evolution of urban environments and air quality control policies, a dataset that concluded nearly a decade ago is no longer representative of current spatiotemporal dynamics. The authors must validate the method on contemporary data prove its practical utility.
The authors limit their analysis to a 6-hour horizon. For effective environmental management and public health planning, a 1-hour to 24-hour horizon (including 12h and 24h steps) is the industry standard to capture full diurnal cycles.
The manuscript fails to compare its results with several high-impact studies that have addressed the same or similar datasets with superior methodological positioning. Specific missing literature includes recent advances in hybrid deep learning and hyperparameter optimization for PM2.5 forecasting. As an example, a comparison can be made with the following studies, identified through an initial literature review:
https://www.sciencedirect.com/science/article/pii/S095741742031157X
https://link.springer.com/article/10.1007/s44230-023-00039-x
https://www.mdpi.com/2073-4433/13/10/1719
https://link.springer.com/article/10.1007/s41060-025-01005-5

The notation for graph types is inconsistent.
Air quality time series are notoriously non-stationary. The manuscript highlights complex spatiotemporal dependencies but lacks a formal discussion or statistical test on how the model handles distribution drift in long-term sequences.
The ensemble task is formulated as a Markov Decision Process. However, the state s_t is restricted to a 24-step historical window. Air quality dynamics often involve long-range dependencies that exceed this window, suggesting that the Markovian property may not hold, potentially leading to sub-optimal policy convergence in the DDPG agent.
The architectural choices (e.g., TCN kernel size, dilation rates) appear heuristic. As established in the literature, meta-heuristic optimization is essential for tuning deep learning models in PM2.5 forecasting to avoid local minima and ensure statistical validity.
The Introduction does not sufficiently differentiate why a dynamic RL ensemble is preferred over multi-task learning architectures, which are standard for multi-station forecasting.
There is no mention of the hardware specifications (CPU/GPU models), numerical precision, or batch sizes.

Due to the use of an outdated dataset, the lack of crucial baselines, and the presence of technical inconsistencies in the RL formulation, this manuscript is not suitable for publication in its current form.

---

## Reviewer #2

### Manuscript Overview

The manuscript presents a technical study focused on the development and evaluation of an AI-based system/model, supported by experimental analysis and performance validation. The topic aligns with the scope of applied artificial intelligence and demonstrates relevance to real-world applications. While the work shows promise, several conceptual, methodological, and reporting-related gaps must be addressed to meet the rigor required for archival publication.

### Strengths

**Relevance and Practical Motivation**
The problem addressed is practically significant and well-aligned with contemporary research trends in applied AI. The manuscript attempts to bridge theoretical modeling with real-world applicability, which is a commendable direction.

**Structured Methodology**
The overall pipeline is logically structured, with clear separation between data acquisition, preprocessing, modeling, and evaluation stages. This modular design improves readability and reproducibility.

**Experimental Effort**
The authors have conducted multiple experiments and reported quantitative results using standard evaluation metrics. The inclusion of comparative baselines reflects an effort toward objective performance validation.

**Clarity of Figures and Workflow Diagrams**
Visual illustrations effectively convey the system architecture and experimental workflow, aiding reader comprehension.

**Potential for Extension**
The proposed framework appears extensible and could serve as a foundation for future improvements or domain-specific adaptations.

### Weaknesses

**Insufficient Technical Novelty Justification**
While the manuscript integrates existing techniques effectively, the degree of algorithmic or methodological novelty is not clearly articulated. The contribution risks being perceived as incremental rather than innovative.

**Limited Theoretical Grounding**
The manuscript relies heavily on empirical performance without sufficient theoretical explanation or justification of key design choices (e.g., model architecture, parameter selection, or optimization strategy).

**Incomplete Comparison with State-of-the-Art**
The comparative analysis lacks coverage of the most recent and relevant state-of-the-art methods. Some baselines appear outdated or insufficiently competitive.

**Overstated Claims**
Certain claims made in the abstract and conclusions (e.g., robustness, efficiency, or generalizability) are not fully supported by experimental evidence or ablation studies.

**Language and Presentation Issues**
The manuscript contains instances of grammatical inconsistency, ambiguous phrasing, and repetitive explanations, which reduce overall clarity and professionalism.

### Technical Limitations

**Dataset Constraints and Generalizability**
The experiments are conducted on a limited dataset or a narrow set of conditions. There is insufficient discussion regarding how well the proposed approach generalizes to unseen environments or real-world deployment scenarios.

**Lack of Ablation Studies**
The contribution of individual components within the proposed framework is not quantitatively analyzed. This makes it difficult to assess which design elements are truly responsible for performance gains.

**Computational Cost and Efficiency**
The manuscript does not provide concrete analysis of computational complexity, training/inference time, or resource requirements, which is particularly important for applied AI systems.

**Reproducibility Concerns**
Key implementation details (e.g., hyperparameters, training protocols, hardware specifications) are insufficiently documented, potentially limiting reproducibility.

**Statistical Significance**
Performance improvements are reported without statistical significance testing, confidence intervals, or variance analysis, weakening the reliability of the conclusions.

### Areas for Technical Improvement

**Clarify and Strengthen Novel Contributions**
The authors should explicitly distinguish their work from closely related methods and clearly articulate what is fundamentally new, both conceptually and technically.

**Improve literature survey by addressing the following recent papers:**
1. Urban air quality index forecasting using multivariate convolutional neural network based customized stacked long short-term memory model
2. Federated learning-based air quality prediction for smart cities using BGRU model
3. Controlling air pollution in data centers using green data centers
4. Bidirectional Long Short-Term Memory for Enhanced Air Quality Index
5. Urban Resilience: Using Autoencoder-Decoder LSTM Model with Green Roofs and Vertical Gardens to Combat Air Pollution
6. Smart Cities' Clean Air: Federated Bidirectional Long Short-Term Memory for Enhanced Air Quality Index Forecasting
7. Temperature-Wave Analysis: A Work with Ensemble Regression Prediction Method

**Expand Experimental Validation**

Include more diverse datasets or cross-domain testing

Add ablation studies to isolate component-wise contributions

Perform robustness testing under noisy or adverse conditions

**Improve Baseline Selection**
Incorporate stronger and more recent baselines, ensuring fair comparison with methods that represent the current state of the art.

**Include Efficiency and Resource Analysis**
Provide quantitative evaluation of computational cost, memory usage, and scalability to support claims of practical applicability.

**Enhance Reproducibility**
Add a detailed implementation section, including hyperparameters, training epochs, optimizer settings, and hardware configuration.

**Refine Presentation and Writing Quality**
A careful language edit is recommended to remove ambiguity, improve flow, and ensure consistency across sections.

### Overall Limitations

Limited novelty relative to existing literature

Restricted experimental scope

Insufficient theoretical justification

Weak reproducibility support

These limitations do not invalidate the work but currently prevent it from reaching the standard required for acceptance without major revision.

### Final Recommendation

Major Revision

The manuscript addresses a relevant problem and demonstrates potential impact. However, substantial improvements are required in terms of technical depth, experimental rigor, novelty clarification, and presentation quality. Addressing the points raised above would significantly strengthen the manuscript and improve its suitability for publication.

---

## Reviewer #3

The conclusion is overly general. It should summarize key results with numbers and avoid making unsupported claims. Future work can be expanded with concrete directions.
Limitations are missing. The paper should explicitly mention dataset constraints, assumptions, potential biases, and situations where the model may fail.
The discussion is superficial. The authors mainly repeat numerical results without explaining why the model performs better or worse. Deeper insights and interpretation are needed.
Some figures are low-quality and lack proper labels. Captions are brief and do not explain the significance of each figure. Tables need better formatting and consistent terminology.
Comparisons are inadequate. Only a few models are compared, and the selection of baselines is not justified. The authors should include strong state-of-the-art models for a meaningful comparison.
Metrics are not well justified. More parameter should be added for a complete evaluation. Tables should be formatted clearly and consistently.
The manuscript needs language polishing. Some sentences are unclear or grammatically incorrect. Technical terms should be used consistently, and sections should flow more logically.
Related Works need to be extend with recent published papers. Some Sample related works suggest to refer.
"Advanced Air Quality Prediction in Metropolitan Delhi via Graph Multi-Attention Network and Bayesian Hyperparameter Optimization," 2025 5th International Conference on Soft Computing for Security Applications (ICSCSA), Salem, India, 2025, pp. 1281-1288, doi: 10.1109/ICSCSA66339.2025.11171159
"STGNN-TCN: Hybrid Model for Spatiotemporal Air Quality Prediction based on Spatio-Temporal Graph Neural Networks and Temporal Convolutional Networks," 2025 Third International Conference on Augmented Intelligence and Sustainable Systems (ICAISS), Trichy, India, 2025, pp. 993-999, doi: 10.1109/ICAISS61471.2025.11042243.
2025. Al-Biruni Earth Radius Optimization for enhanced environmental data analysis in remote sensing imagery. Agrociencia, pp.1-18.
Machine vision hypergraph neural networks for early detection of damping-off and root rot disease in coffee plantations, Agrociencia, 1-22
Blockchain-Based crop monitoring using an interplanetary file system, Agrociencia, 1-15, 2025

No acronyms may be used in the title.

No acronyms may be used in the keywords.

Acronyms in the abstract must be defined on first usage.

Both the implemented AI, as well as the application of AI must be mentioned in the keywords and abstract

These rules are applicable to all acronyms. This includes commonly used acronyms (e.g. AI, ML, 3D), model names (e.g. YOLO, swin transformer, BERT), dataset names (e.g. KITTI, CEC2017, PASCAL VOC), units (e.g. mm, MPa, GFlops), algorithm names (e.g. PSO, LASSO), and novel proposals (e.g. names of new architectures)

The title, keywords, and abstract entered on the EM system must match the corresponding entries in the manuscript; care should be taken to ensure that no equation references appear incorrectly on the EM system.
