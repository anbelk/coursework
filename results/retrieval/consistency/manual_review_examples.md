# Manual review examples for author_retriever

These examples are selected from stored LLM ratings: one bad top-5, one mixed top-5, and two strong top-5 cases. They are not a replacement for a blind human audit.

## User `A5062977994`: Michael Boratko
Top-5 candidates with LLM score >= 2: `0/5`.

Recent user papers:
- 2021: Min/Max Stability and Box Distributions
- 2021: Box Embeddings: An open-source library for representation learning using geometric structures
- 2022: An Evaluative Measure of Clustering Methods Incorporating Hyperparameter Sensitivity
- 2022: Word2Box: Capturing Set-Theoretic Semantics of Words using Box Embeddings
- 2024: Every Answer Matters: Evaluating Commonsense with Probabilistic Measures

Top-5 candidates:
- rank 1, score 0, `A5104143781` Xiaofei Ma, cosine 0.715
  - reason: The user focuses on geometric and probabilistic representation learning, clustering evaluation, and commonsense evaluation, while the candidate's work centers on language model training techniques, explainability, continual learning, dialogue summarization, and reranking. There is no concrete shared task, method, dataset, or evaluation problem linking...
  - candidate paper: 2023: Exploring Continual Learning for Code Generation Models
  - candidate paper: 2023: SWING: Balancing Coverage and Faithfulness for Dialogue Summarization
  - candidate paper: 2024: Lightweight reranking for language model generations
- rank 2, score 0, `A5060347694` Dani Yogatama, cosine 0.708
  - reason: The user's work focuses on geometric and probabilistic box embeddings and clustering evaluation, while the candidate's work centers on memory-augmented language models, dense retrieval, and language model pretraining analysis. There is no concrete shared task, method, dataset, or evaluation problem linking their research.
  - candidate paper: 2023: Scaling Laws vs Model Architectures: How does Inductive Bias Influence Scaling?
  - candidate paper: 2023: The Distributional Hypothesis Does Not Fully Explain the Benefits of Masked Language Model Pretraining
  - candidate paper: 2024: On Retrieval Augmentation and the Limitations of Language Model Training
- rank 3, score 0, `A5032039796` Emmanouil Antonios Platanios, cosine 0.693
  - reason: User's work focuses on geometric and probabilistic representations (box embeddings) and clustering evaluation, while candidate's work centers on semantic parsing, relation extraction, and dataset quality in NLP, with no concrete shared tasks, methods, or datasets.
  - candidate paper: 2022: Guided K-best Selection for Semantic Parsing Annotation
  - candidate paper: 2022: Online Semantic Parsing for Latency Reduction in Task-Oriented Dialogue
  - candidate paper: 2022: When More Data Hurts: A Troubling Quirk in Developing Broad-Coverage Natural Language Understanding Systems
- rank 4, score 0, `A5075728689` Jinfeng Rao, cosine 0.690
  - reason: User's work focuses on geometric and probabilistic representation learning (box embeddings) and clustering evaluation, while candidate's work centers on neural natural language generation and model scaling; no concrete shared tasks, methods, or datasets identified.
  - candidate paper: 2023: Scaling Laws vs Model Architectures: How does Inductive Bias Influence Scaling?
  - candidate paper: 2023: DSI++: Updating Transformer Memory with New Documents
  - candidate paper: 2023: Transcending Scaling Laws with 0.1% Extra Compute
- rank 5, score 0, `A5067997502` Yashar Mehdad, cosine 0.685
  - reason: The user's work focuses on geometric and probabilistic box embeddings and clustering evaluation, while the candidate's work centers on dense retrieval, long text sequence modeling, audio language models, and quantization for large language models. There is no concrete shared task, method, dataset, or evaluation problem linking their research.
  - candidate paper: 2023: Adapting Pretrained Text-to-Text Models for Long Text Sequences
  - candidate paper: 2024: Attention or Convolution: Transformer Encoders in Audio Language Models for Inference Efficiency
  - candidate paper: 2024: LLM-QAT: Data-Free Quantization Aware Training for Large Language Models

## User `A5023724461`: Dan Zeng
Top-5 candidates with LLM score >= 2: `2/5`.

Recent user papers:
- 2023: Model Conversion via Differentially Private Data-Free Distillation
- 2023: Normal Image Guided Segmentation Framework for Unsupervised Anomaly Detection
- 2023: Personalized Federated Learning via Backbone Self-Distillation
- 2024: Cross-Modal Quantization for Co-Speech Gesture Generation
- 2024: Densely Connected Transformer with Frequency Awareness and Sam Guidance for Semi-Supervised Hyperspectral Image Classification

Top-5 candidates:
- rank 1, score 2, `A5060042752` Bo Du, cosine 0.791
  - reason: Both authors work on knowledge distillation and model compression techniques, with the user focusing on differentially private data-free distillation and personalized federated learning via self-distillation, while the candidate explores knowledge distillation for autoregressive language models and semi-supervised learning. Their expertise in model...
  - candidate paper: 2024: Learning a generalizable re-identification model from unlabelled data with domain-agnostic expert
  - candidate paper: 2024: Training-Free Robust Neural Network Search Via Pruning
  - candidate paper: 2024: Learning from Imperfect Data: Towards Efficient Knowledge Distillation of Autoregressive Language Models for Text-to-SQL
- rank 2, score 0, `A5100404947` Jie Yang, cosine 0.789
  - reason: The candidate's work focuses on domain adaptation, adversarial robustness of quantized neural networks, semi-supervised learning with class mismatch, and language model fine-tuning for debate generation, which do not concretely overlap with the user's research on differentially private data-free distillation, unsupervised anomaly detection, personalized...
  - candidate paper: 2023: Dynamic Weighted Adversarial Learning for Semi-Supervised Classification under Intersectional Class Mismatch
  - candidate paper: 2023: Robot Debater: Debate-styled Text Auto-generation System Based on Large Foundation Language Models
  - candidate paper: 2024: Feature Transformation Based on Autoencoder to Oversample on Imbalanced Data
- rank 3, score 0, `A5100668696` Jiashi Feng, cosine 0.787
  - reason: The user's recent work focuses on privacy-preserving model distillation, anomaly detection, federated learning, cross-modal gesture generation, and hyperspectral image classification, while the candidate's work centers on domain adaptation, class incremental learning, long-tailed learning, geological exploration, and LLM-powered multi-agent systems. There...
  - candidate paper: 2023: Deep Long-Tailed Learning: A Survey
  - candidate paper: 2024: Granite Rock Mass Identification and Geothermal Well Location Deployment Based on the Wide Field Electromagnetic Method: A Case Study of Sanshui Basin
  - candidate paper: 2024: MAgIC: Investigation of Large Language Model Powered Multi-Agent in Cognition, Adaptability, Rationality and Collaboration
- rank 4, score 2, `A5014346487` Ye Yuan, cosine 0.786
  - reason: The user focuses on privacy-preserving model conversion, federated learning, and anomaly detection, while the candidate works on data scarcity mitigation, black-box model attribute reverse engineering, and adversarial attacks on segmentation models like SAM. Both have expertise in model robustness, privacy, and data-efficient learning, which are...
  - candidate paper: 2024: DREAM: Domain-Agnostic Reverse Engineering Attributes of Black-Box Model
  - candidate paper: 2024: Cross-Point Adversarial Attack Based on Feature Neighborhood Disruption Against Segment Anything Model
  - candidate paper: 2024: Importance-aware Shared Parameter Subspace Learning for Domain Incremental Learning
- rank 5, score 0, `A5101917144` Jun Zhou, cosine 0.786
  - reason: The user's recent work focuses on privacy-preserving model distillation, anomaly detection, federated learning, cross-modal gesture generation, and hyperspectral image classification, while the candidate's work centers on analog circuit fault diagnosis, adversarial patches in face recognition, semi-supervised learning for tabular data, probabilistic...
  - candidate paper: 2022: Semi-Supervised Learning with Data Augmentation for Tabular Data
  - candidate paper: 2023: A distribution-free method for probabilistic prediction
  - candidate paper: 2024: Multi-Branch Instance Segmentation of Cervical Cells

## User `A5013050263`: Shinnosuke Takamichi
Top-5 candidates with LLM score >= 2: `5/5`.

Recent user papers:
- 2024: Do Learned Speech Symbols Follow Zipf’s Law?
- 2024: Diversity-Based Core-Set Selection for Text-to-Speech with Linguistic and Acoustic Features
- 2024: DNN-Based Ensemble Singing Voice Synthesis With Interactions Between Singers
- 2024: Real-Time Noise Estimation for Lombard-Effect Speech Synthesis in Human–Avatar Dialogue Systems
- 2024: NecoBERT: Self-Supervised Learning Model Trained by Masked Language Modeling on Rich Acoustic Features Derived from Neural Audio Codec

Top-5 candidates:
- rank 1, score 3, `A5035532752` Chenpeng Du, cosine 0.884
  - reason: Both authors focus on discrete speech tokens and text-to-speech (TTS) synthesis, with overlapping interests in learned speech symbols, discrete speech units, and TTS system improvements. The candidate's work on universal speech discrete tokens and unified context-aware TTS frameworks complements the user's research on learned speech symbols, core-set...
  - candidate paper: 2024: E$^{3}$TTS: End-to-End Text-Based Speech Editing TTS System and Its Applications
  - candidate paper: 2024: The X-Lance Technical Report for Interspeech 2024 Speech Processing using Discrete Speech Unit Challenge
  - candidate paper: 2024: Attention-Constrained Inference For Robust Decoder-Only Text-to-Speech
- rank 2, score 2, `A5073918837` Ruibo Fu, cosine 0.880
  - reason: Both authors work on speech synthesis and speech representation learning. The user focuses on learned speech symbols, TTS corpus selection, and speech feature extraction (e.g., NecoBERT), while the candidate works on minimally-supervised TTS with semantic coding and speech representation learning bridging text and acoustic information. Their shared...
  - candidate paper: 2024: EELE: Exploring Efficient and Extensible LoRA Integration in Emotional Text-to-Speech
  - candidate paper: 2024: Transferring Personality Knowledge to Multimodal Sentiment Analysis
  - candidate paper: 2024: Personality-Guided Multimodal Sentiment Analysis
- rank 3, score 2, `A5101736420` Shan Yang, cosine 0.866
  - reason: Both authors work on speech synthesis and voice conversion technologies, with the user focusing on TTS, singing voice synthesis, and speech feature learning, and the candidate working on zero-shot TTS, voice conversion, and acoustic modeling. Their expertise in speech generation and acoustic modeling is complementary and relevant for collaboration.
  - candidate paper: 2022: End-to-End Voice Conversion with Information Perturbation
  - candidate paper: 2024: Heuristic-Driven, Type-Specific Embedding in Parallel Spaces for Enhancing Knowledge Graph Reasoning
  - candidate paper: 2024: Unleashing the Power of Large Language Models in Zero-shot Relation Extraction via Self-Prompting
- rank 4, score 3, `A5101522530` Xu Tan, cosine 0.855
  - reason: Both authors focus on speech synthesis and singing voice synthesis using advanced deep learning models. The user works on TTS, singing voice synthesis, and speech feature learning, while the candidate develops state-of-the-art speech and singing voice synthesis methods including diffusion and consistency models, which strongly complement the user's...
  - candidate paper: 2024: Contrastive Context-Speech Pretraining for Expressive Text-to-Speech Synthesis
  - candidate paper: 2024: FlashSpeech: Efficient Zero-Shot Speech Synthesis
  - candidate paper: 2024: COMOSVC: Consistency Model-Based Singing Voice Conversion
- rank 5, score 3, `A5052468556` Hoon Young Cho, cosine 0.851
  - reason: Both authors focus on speech synthesis, including singing voice synthesis and text-to-speech systems. The user has worked on ensemble singing voice synthesis and TTS corpus selection, while the candidate has developed non-autoregressive Korean singing voice synthesis and hierarchical context-aware TTS models. Their work shares strong concrete matches in...
  - candidate paper: 2021: FastPitchFormant: Source-Filter Based Decomposed Modeling for Speech Synthesis
  - candidate paper: 2024: Latent Filling: Latent Space Data Augmentation for Zero-Shot Speech Synthesis
  - candidate paper: 2024: Mels-Tts : Multi-Emotion Multi-Lingual Multi-Speaker Text-To-Speech System Via Disentangled Style Tokens

## User `A5001299811`: Mounir Hamdi
Top-5 candidates with LLM score >= 2: `5/5`.

Recent user papers:
- 2023: Optimal Resource Management for Hierarchical Federated Learning Over HetNets With Wireless Energy Transfer
- 2023: Dynamic Pruning for Distributed Inference via Explainable AI: A Healthcare Use Case
- 2024: Reinforcement learning-based dynamic pruning for distributed inference via explainable AI in healthcare IoT systems
- 2024: A Blockchain-Based Reliable Federated Meta-Learning for Metaverse: A Dual Game Framework
- 2024: Multi-agent reinforcement learning for privacy-aware distributed CNN in heterogeneous IoT surveillance systems

Top-5 candidates:
- rank 1, score 2, `A5081182489` Xi Zheng, cosine 0.830
  - reason: The user focuses on distributed and federated learning, resource management, and privacy-aware deep learning in IoT and healthcare systems, while the candidate works on testing and reliability of learning-enabled cyber-physical systems, neuro-symbolic AI for AIoT, and decentralized learning incentives. Their expertise complements each other in distributed...
  - candidate paper: 2024: Distributed Learning in Intelligent Transportation Systems: A Survey
  - candidate paper: 2024: iDOL: Incentivized Decentralized Opportunistic Learning
  - candidate paper: 2024: Prompt Engineering Adversarial Attack Against Image Captioning Models
- rank 2, score 2, `A5037865550` M. Shamim Hossain, cosine 0.824
  - reason: Both authors work on federated learning and IoT applications, with the user focusing on hierarchical federated learning and distributed inference in healthcare and industrial IoT, while the candidate addresses federated learning optimization, fairness, energy efficiency, and self-supervised learning in IoT and industrial contexts, providing complementary...
  - candidate paper: 2024: An effective Federated Learning system for Industrial IoT data streaming
  - candidate paper: 2024: Federated Self-Supervised Learning Based on Prototypes Clustering Contrastive Learning for Internet of Vehicles Applications
  - candidate paper: 2024: A Data Completion Algorithm Based on Low-Rank Prior Knowledge for Data-Driven Applications
- rank 3, score 2, `A5005228053` Shui Yu, cosine 0.811
  - reason: Both authors work on federated learning and distributed AI in IoT contexts, with the user focusing on hierarchical federated learning, distributed inference, and reinforcement learning for resource-constrained devices, while the candidate focuses on hierarchical federated learning improvements, federated unlearning, and privacy in federated learning....
  - candidate paper: 2024: FedU: Federated Unlearning via User-Side Influence Approximation Forgetting
  - candidate paper: 2024: Prompt Engineering Adversarial Attack Against Image Captioning Models
  - candidate paper: 2024: From the Perspective of AI Safety: Analyzing the Impact of XAI Performance on Adversarial Attack
- rank 4, score 2, `A5072627238` Zhipeng Cai, cosine 0.798
  - reason: Both authors work on federated learning and distributed learning in IoT and edge settings, with the user focusing on resource management, pruning, and privacy in healthcare and industrial IoT, while the candidate focuses on federated learning challenges such as non-IID data, machine unlearning, and streaming federated learning. Their expertise complements...
  - candidate paper: 2024: GANFed: GAN-Based Federated Learning with Non-IID Datasets in Edge IoTs
  - candidate paper: 2024: Federating from History in Streaming Federated Learning
  - candidate paper: 2024: Spectrum Prediction via Graph Structure Learning
- rank 5, score 2, `A5100429898` Qiang Li, cosine 0.798
  - reason: The user focuses on federated learning, distributed inference, and privacy-aware deep learning in IoT and healthcare systems, while the candidate works on security aspects in federated learning such as poisoning and backdoor attacks, and blockchain-based authentication in IoT. Their expertise complements each other in securing and optimizing federated...
  - candidate paper: 2024: Breaking State-of-the-Art Poisoning Defenses to Federated Learning: An Optimization-Based Attack Framework
  - candidate paper: 2024: Distributed Backdoor Attacks on Federated Graph Learning and Certified Defenses
  - candidate paper: 2024: Variational Learning of Integrated Quantum Photonic Circuits via Genetic Algorithm
