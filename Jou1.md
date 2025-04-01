## **🕐 Jour 1 :Comprendre l’Architecture des Solutions IA**  
🎯 **Objectif** : Apprendre **comment structurer un projet IA de bout en bout** en maîtrisant les composants clés.  

### **📌 1. Les 3 piliers d’une solution IA**  

Un projet IA repose sur **3 composants fondamentaux** :  

| Composant   | Rôle principal | Exemples |
|------------|--------------|----------|
| **Données** | Stockage, structuration et prétraitement des données pour l’IA | Bases relationnelles (PostgreSQL), Data Lakes (S3), Vector DB (Pinecone, ChromaDB) |
| **Modèles** | Sélection, entraînement et optimisation des modèles LLM | OpenAI GPT-4, Mistral, LLaMA, Claude |
| **Infrastructure** | Orchestration, scalabilité et déploiement des modèles | Kubernetes, API Gateway, Serverless GPUs (RunPod, Modal) |

🔗 **Références utiles** :  
- [📖 Introduction to AI System Design](https://course.fast.ai/) (FastAI)  
- [📘 Machine Learning Systems Design (Chip Huyen)](https://huyenchip.com/ml-sys-design/toc.html)  
- [📑 AI Engineering Guide (Cohere)](https://cohere.com/blog/ai-engineering-guide)  

---

### **📌 2. Architecture LLMOps : Du Prototype à la Production**  

LLMOps est une approche pour **déployer et gérer des modèles IA en production** (similaire à MLOps).  

🚀 **Pipeline IA optimal** :  
1. **Data Ingestion** → Collecte et nettoyage des données  
2. **Vectorization** → Transformation en embeddings (FAISS, Pinecone)  
3. **Model Serving** → API LLM (OpenAI, vLLM, TGI)  
4. **Application** → Interface utilisateur et intégration  

📌 **Références** :  
- [📖 MLOps Primer – Google](https://developers.google.com/machine-learning/mlops)  
- [📘 Designing Machine Learning Systems (Chip Huyen)](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)  

---

### **📌 3. Outils clés pour structurer un projet IA**  

| Outil | Usage | Lien |
|------|-------|------|
| **Langchain** | Orchestration et interaction avec LLMs | [Langchain Docs](https://python.langchain.com/) |
| **Langraph** | Workflows IA optimisés | [Langraph Docs](https://langraph.dev/) |
| **Pinecone** | Base de données vectorielle | [Pinecone Docs](https://docs.pinecone.io/) |
| **vLLM** | Exécution rapide de LLMs | [vLLM GitHub](https://github.com/vllm-project/vllm) |
| **TGI (Text Generation Inference)** | Déploiement scalable de LLMs | [TGI GitHub](https://github.com/huggingface/text-generation-inference) |

🔗 **Ressources supplémentaires** :  
- [📖 Architecture for AI-Driven Applications (Hugging Face)](https://huggingface.co/blog/ai-apps)  
- [📑 AI Infrastructure at Scale (Meta)](https://engineering.fb.com/2021/11/16/open-source/ai-infra-scale/)  

---

### **📌 4. Exercice pratique (30 min) : Cartographier un projet IA**  

✅ **Défi** : Prends un projet IA fictif (ex. un chatbot avec mémoire) et décris :  
1. **Les données requises** (textes, logs, etc.)  
2. **Le modèle utilisé** (GPT-4, Mistral, etc.)  
3. **L’infrastructure** (API, Kubernetes, Vector DB)  

Tu peux t’inspirer de ce schéma standard :  
![Architecture IA](https://huggingface.co/blog/assets/50_llm_app/llm_architecture.png)  

---

Après cette première heure, tu auras une **vue claire sur l’architecture IA** et les outils essentiels.  

➡️ **Prêt pour la suite ? On passe à la structuration des données !** 🚀
