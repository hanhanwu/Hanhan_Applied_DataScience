# AI Learning Notes

## 🍀Promising to Explore

### Synthetic Data
* [SynthLLM: Generate Synthetic Q&A with LLM and graph][66]
* [Simulate Social Behaviours with Agents][68]
  * [The paper][69]
 
### Knowledge Graph
* [FalkorDB: Build Fast Knowledge Graph][89]
  * https://github.com/FalkorDB/FalkorDB 
* [Financial KG][73]
  * [The KG this paper has built][72], the load this file into KG tools to extract knowledge

### Prompt Engineering & Prompt Strategy
* [Prompt methods with reasoning][58]
* [Verbalized Sampling - A prompting strategy to improve LLM output's diversity][87]
  * [The paper][88] 

### Context Engineering Solutions
* [TOON Format for Token Efficiency][86], can be used in context
* [ARQ (Attentive Reasoning Queries)][78]
  * It creates structured context ("guidelines" or "journey"), selecting the most relevant context, add into the prompt and execute at the step where needs the context
* [AEC (Agentic Context Engine)][80], [the paper][79]
  * ACE works by maintaining a structured, evolving playbook of discrete knowledge "bullets", selecting the most relevant ones for each task, feeding them to the LLM as context, and updating the playbook over time based on task feedback.
  * We can imagine each "bullets" is like a JSON file that stores the context and the meta data

### Document Parsing
* [Docling][42]

### Indexing
* [Different indexing methods for RAG][81]

### Embedding
* [Hugging Face Embedding Models Leaderboard to Find Top Embedding Models][85]
* [Google Deep Mind EmbeddingGemma][76] - Small, Fast, on-device/offline, supports multilingual embedding.
* [Contextualized Chunk Embeddings][74]
  * [Performance comparison with a few other embedding models][75] 
* [Model2Vec: Faster & Lighter Embedding][55]
  * Static Embedding method, which means each token has fixed representation and therefore not context aware. This is also why this type of model is fast.
  * Because it's not context aware, can be less accurate 
* [ColBERT][26]
  * Enables more effective retrieval than bi-encoder, more efficient than cross-encoder
  * ColBERT uses the MaxSim operation, which computes the maximum semantic similarity between query and document token embeddings. This allows for <b>fine-grained matching at the token level</b>, leading to better relevance estimation, especially <b>in cases where specific terms in the query are critical for retrieval</b>.
  * [How to use ColBERT in RAG][43]

### RAG Alternatives
* [CAG (Cache-augmented generation), a faster alternative for RAG][19]
  * [RAG + CAG][82]: RAG retrieval & CAG's KV-cache layer (for faster reusable memory), letting the model carry forward context, reasoning traces, and prior outputs.
* [Zep][36]
  * "Build AI agents that continually learn. Power personalized experiences."
  * It uses temporal knowledge graph

### RAG Techniques
* [RAG Techniques][18]
* [LangChain bRAG][39]
  * [Query Translation][40] 💖
    * <b>Multi Query</b>: improve output info coverage, handle ambiguous queries better  --> Enable Diverse Information Coverage
    * <b>RAG Fusion</b>: multiple retrieval from different retrievers, index variations, or different ranking strategies --> Improved Retrieval Ranking
    * <b>Query Decomposition</b>: breakdown complex queries into sub questions --> answer recursively or individually
    * <b>Step Back Query</b>: generate higher level query to emphasize a bigger picture better
    * <b>HyDE</b>: generate hypothetical document for better retrieval
  * [Routing][41]: the process of dynamically directing a query to the most appropriate retriever, database, or model based on its content
  * [Cohere Rerank to improve retrieval relevancy for the query][44] 🌟
    * [How to use Cohere Rerank in Langchain][45]
    * Will be helpful to get the reranked document relevancy score
* [ColiVara - Improve Retrieval without Chunking][35]
  * It treats docs as images and uses vision models for embedding, just like a human would do. This provides far better accuracy than traditional RAG systems that rely on chunking, without latency issues. 
* Text Chunking
  * [Explain how does chunking work][22] 
  * [LangChain's Text Spliter][13] vs [Chonkie][14]
  * [Langchain has more options for embeddings, retrievers, etc.][16]
* [Extend RAG with RCA (Root Cause Analysis)][57] 🌟
  * Input error code --> causal knowledge graph --> output causes
  * To build causal knowledge graph (RAG & RCA):
    * Retriever: Retrieves documents such as failure reports, maintenance logs, expert knowledge, etc.
    * Generator: Reads the retrieved text and generates structured causal relationships.
    * These relationships are added to the causal knowledge graph.
  * If input can't find knowledge in the causal knowledge graph, go through RAG & RCA process.
  * [How did the KG was built][65], hypergraph

### Agents
* [Microsoft Agent Lightning][83]
  * Build with ANY agent framework
  * Selectively optimize one or more agents in a multi-agent system
  * Embraces Algorithms like Reinforcement Learning, Automatic Prompt Optimization, Supervised Fine-tuning and more
* [ART - Agent Reinforcement Trainer][84]
  * ART is an open-source RL framework that improves agent reliability by allowing LLMs to learn from experience.  
* [Tips for Context Engineering from Langchain][67]
  * Write context
  * Select context
  * Compress context
  * Isolate context 
* [Browser Use - Allow agents to interact with the browser][23]
* [Agentic AI Design Patterns][27]
* [FastAPI Langgraph Agent Production Ready Code Template][51]
* [Build MCP Agents with Langchain][54]

### Architecture
* [How does Vector Database work][59]
  * [The VectorDB improved key words search's efficiency and effectiveness][62] 
* [syftr: Find Optimal Agentic Workflow][61]
* [FastAPI & MCP & LangGraph Template][53]
* <b>GCP</b>
  * [GCP multi-agent systems][60]

### Evaluation
* [Beyond A/B Test - Different production tests methods for model performance][70]
* [Trustworthy Language Model (TLM)][20], [reduce Hallucinations][21] for any LLM
  * [Use TLM in LangChain][25]
* [VertaxAI Search: grounding for Gemini RAG][33]
  * Check the video, it shows how could you get python code for this
* [Opik: trace RAG workflow & evaluate LLM apps][34]

### Open Source RAGs
* [RAG-Anything][77]: all-in-one context-aware multimodal processing 
* [LangConnect-Client][63]
  * [LangConnect][64] 

### LLM Model Fine Tuning
* [Transformer Lab: Experiment with Fine Tuning LLM Models][52]
* [How to fine tune DeepSeek R1][30]
  * R1 is reasoning model 
  * [the completed code][29]
* [Reinforcement Fine-tuning of LLMs][49]
  * It uses an online reward approach so you don't need any static labels at the start
  * [Example code][50]
 
### AI Application Deployment
* [Beta9 - Deploy AI Application in Serverless Manner][71]
* [Ray Serve: Deploy AI application with scale][31]
  * [Ray Serve with Langchain][32]
  * [Ray Serve LLM][47]
    * [Models not supported by vLLM can use OpenAI compatable endpoint][48] 
    
### 0 Code AI Developers 💖
* [Srcbook - Prompt to build an app][46], the app is written in TypeScript, you can also edit the generated code, supports both web and mobile app
* [Khoj - A Few Clicks to Create Your Own Agents][15]

### AI Models
* [Mistral AI Classifiers][56]
  * "Utilize our small yet highly efficient models and training methods to develop custom classifiers for moderation, intent detection, sentiment analysis, data clustering, fraud detection, spam filtering, recommendation systems, etc."

### LLM Visualization
* [Visually to understand LLM models in detail][28] 💖


## 🍀LangChain Universe 😉
* [LangChain's RAG chatbot][24], much better than its tutorials, it allows to use different LLMs and can write code for you! 💖
* [LangSmith Cookbook][8], different ways to test & evaluate & optimization
  * LangServer is replaced by LangGraph
* [Prompts in LangChain Hub][12]
* [Langchain MCP adapters][37]
  * [MCP][38]: The Model Context Protocol (MCP) lets you build servers that expose data and functionality to LLM applications in a secure, standardized way. Think of it like a web API, but specifically designed for LLM interactions.

### LangGraph
* [LangGraph github][9]: build agent workflows
* [LangGraph Basics][17]: how to use Time Travel, Human in the Loop, etc.
* [LangGraph exercise][11] 💖

### Strategy
* The content here can be more political...
* [Google Whitepaper: POC to Production][90], more for GCP to build AI agent


[1]:https://github.com/run-llama/llamacloud-demo/blob/main/examples/10k_apple_tesla/demo_file_retrieval.ipynb
[2]:https://github.com/run-llama/llamacloud-demo/tree/main/examples
[3]:https://github.com/run-llama/llamacloud-demo/blob/main/examples/10k_apple_tesla/demo_subquestion.ipynb
[4]:https://github.com/run-llama/llamacloud-demo/blob/main/examples/10k_apple_tesla/demo_ensemble_retrieval.ipynb
[5]:https://github.com/run-llama/llamacloud-demo/blob/main/examples/advanced_rag/auto_retrieval_img.png
[6]:https://github.com/run-llama/llamacloud-demo/blob/main/examples/advanced_rag/auto_retrieval.ipynb
[7]:https://github.com/run-llama/llamacloud-demo/blob/main/examples/advanced_rag/corrective_rag_workflow.ipynb
[8]:https://github.com/langchain-ai/langsmith-cookbook/tree/main
[9]:https://github.com/langchain-ai/langgraph
[10]:https://www.datacamp.com/blog/langchain-vs-llamaindex?utm_source=google&utm_medium=paid_search&utm_campaignid=19589720821&utm_adgroupid=152984010854&utm_device=c&utm_keyword=&utm_matchtype=&utm_network=g&utm_adpostion=&utm_creative=724847709973&utm_targetid=dsa-2222697810678&utm_loc_interest_ms=&utm_loc_physical_ms=9000960&utm_content=DSA~blog~Artificial-Intelligence&utm_campaign=230119_1-sea~dsa~tofu_2-b2c_3-row-p1_4-prc_5-na_6-na_7-le_8-pdsh-go_9-nb-e_10-na_11-na-dec24&gad_source=1&gclid=CjwKCAiAjp-7BhBZEiwAmh9rBQQoaxAXWDeMMQY5qKPVmH3n3s_j-VdMZJiW_yWJPysrdFyReTQIxRoCU7MQAvD_BwE
[11]:https://github.com/hanhanwu/Hanhan_LangGraph_Exercise
[12]:https://smith.langchain.com/hub/
[13]:https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/
[14]:https://docs.chonkie.ai/chunkers/overview
[15]:https://app.khoj.dev/agents
[16]:https://python.langchain.com/docs/integrations/text_embedding/
[17]:https://langchain-ai.github.io/langgraph/tutorials/introduction/
[18]:https://github.com/NirDiamant/RAG_Techniques
[19]:https://medium.com/@ronantech/cache-augmented-generation-cag-in-llms-a-step-by-step-tutorial-6ac35d415eec
[20]:https://cleanlab.ai/tlm/
[21]:https://cleanlab.ai/blog/simpleqa/
[22]:https://www.linkedin.com/posts/avi-chawla_5-chunking-strategies-for-rag-explained-in-activity-7283052020809277441-OxZo?utm_source=share&utm_medium=member_desktop
[23]:https://docs.browser-use.com/introduction
[24]:https://chat.langchain.com/
[25]:https://help.cleanlab.ai/tlm/use-cases/tlm_langchain/
[26]:https://github.com/stanford-futuredata/ColBERT
[27]:https://www.linkedin.com/posts/avi-chawla_5-%F0%9D%97%BA%F0%9D%97%BC%F0%9D%98%80%F0%9D%98%81-%F0%9D%97%BD%F0%9D%97%BC%F0%9D%97%BD%F0%9D%98%82%F0%9D%97%B9%F0%9D%97%AE%F0%9D%97%BF-%F0%9D%97%94%F0%9D%97%B4%F0%9D%97%B2%F0%9D%97%BB%F0%9D%98%81%F0%9D%97%B6%F0%9D%97%B0-activity-7288083231944388608-V-iW/?utm_source=share&utm_medium=member_desktop
[28]:https://bbycroft.net/llm
[29]:https://github.com/patchy631/ai-engineering-hub/blob/main/DeepSeek-finetuning/Fine_tune_DeepSeek.ipynb
[30]:https://www.linkedin.com/posts/akshay-pachaar_step-by-step-deepseek-r1-distilled-llama-activity-7289954078418182144-lPIG?utm_source=share&utm_medium=member_desktop
[31]:https://docs.ray.io/en/latest/serve/index.html
[32]:https://python.langchain.com/docs/integrations/providers/ray_serve/
[33]:https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/overview
[34]:https://github.com/comet-ml/opik
[35]:https://github.com/tjmlabs/ColiVara
[36]:https://github.com/getzep/zep?tab=readme-ov-file#examples
[37]:https://github.com/langchain-ai/langchain-mcp-adapters
[38]:https://github.com/modelcontextprotocol/python-sdk
[39]:https://github.com/bRAGAI/bRAG-langchain/
[40]:https://github.com/bRAGAI/bRAG-langchain/blob/main/notebooks/%5B2%5D_rag_with_multi_query.ipynb
[41]:https://github.com/bRAGAI/bRAG-langchain/blob/main/notebooks/%5B3%5D_rag_routing_and_query_construction.ipynb
[42]:https://github.com/DS4SD/docling
[43]:https://github.com/bRAGAI/bRAG-langchain/blob/main/notebooks/%5B4%5D_rag_indexing_and_advanced_retrieval.ipynb
[44]:https://cohere.com/blog/rerank
[45]:https://github.com/bRAGAI/bRAG-langchain/blob/main/notebooks/%5B5%5D_rag_retrieval_and_reranking.ipynb
[46]:https://srcbook.com/
[47]:https://www.anyscale.com/blog/llm-apis-ray-data-serve
[48]:https://docs.ray.io/en/latest/data/working-with-llms.html#batch-inference-with-an-openai-compatible-endpoint
[49]:https://www.linkedin.com/posts/avi-chawla_supervised-reinforcement-fine-tuning-in-activity-7316413686955290624-LUfi?utm_source=share&utm_medium=member_desktop&rcm=ACoAABUa5xMBAWvx7L2IKhfsBuLjhTEWJhTYoNk
[50]:https://colab.research.google.com/drive/1bURdkV_StTbTsYgisUPHaHJqeAr-hMGz?usp=sharing
[51]:https://github.com/wassim249/fastapi-langgraph-agent-production-ready-template
[52]:https://github.com/transformerlab/transformerlab-app
[53]:https://github.com/NicholasGoh/fastapi-mcp-langgraph-template
[54]:https://composio.dev/blog/langchain-mcp-adapter-a-step-by-step-guide-to-build-mcp-agents/
[55]:https://github.com/MinishLab/model2vec
[56]:https://www.linkedin.com/posts/mistralai_classifier-factory-mistral-ai-large-language-activity-7318357052580339714-kXSz?utm_source=share&utm_medium=member_desktop&rcm=ACoAABUa5xMBAWvx7L2IKhfsBuLjhTEWJhTYoNk
[57]:https://blog.fltech.dev/entry/2025/04/16/rca-plm-ja
[58]:https://www.linkedin.com/posts/akshay-pachaar_3-prompting-techniques-for-reasoning-in-llms-activity-7333793371729076224-vMkF?utm_source=share&utm_medium=member_desktop&rcm=ACoAABUa5xMBAWvx7L2IKhfsBuLjhTEWJhTYoNk
[59]:https://qdrant.tech/articles/what-is-a-vector-database/
[60]:https://cloud.google.com/blog/products/ai-machine-learning/build-and-manage-multi-system-agents-with-vertex-ai
[61]:https://github.com/datarobot/syftr
[62]:https://github.com/weaviate/weaviate
[63]:https://github.com/teddynote-lab/LangConnect-Client
[64]:https://github.com/langchain-ai/langconnect
[65]:https://blog-en.fltech.dev/entry/2025/06/02/kgla-en
[66]:https://www.linkedin.com/posts/activity-7351026313387782144-fGJ2?utm_source=share&utm_medium=member_desktop&rcm=ACoAABUa5xMBAWvx7L2IKhfsBuLjhTEWJhTYoNk
[67]:https://blog.langchain.com/context-engineering-for-agents/
[68]:https://github.com/tsinghua-fib-lab/agentsociety/
[69]:https://aclanthology.org/2025.acl-industry.94.pdf
[70]:https://www.linkedin.com/posts/avi-chawla_4-ways-to-test-ml-models-in-production-activity-7357713588267991042-kGq8?utm_source=share&utm_medium=member_desktop&rcm=ACoAABUa5xMBAWvx7L2IKhfsBuLjhTEWJhTYoNk
[71]:https://github.com/beam-cloud/beta9
[72]:https://anonymous.4open.science/r/KG-Financial-Datasets-SP-100-529B/README.md
[73]:https://arxiv.org/pdf/2508.17906
[74]:https://docs.voyageai.com/docs/contextualized-chunk-embeddings#quickstart
[75]:https://blog.voyageai.com/2025/07/23/voyage-context-3/
[76]:https://developers.googleblog.com/en/introducing-embeddinggemma/
[77]:https://github.com/HKUDS/RAG-Anything
[78]:https://www.linkedin.com/posts/activity-7386157423780216832-UGme/?utm_source=share&utm_medium=member_desktop&rcm=ACoAABUa5xMBAWvx7L2IKhfsBuLjhTEWJhTYoNk
[79]:https://arxiv.org/pdf/2510.04618
[80]:https://arxiv.org/pdf/2510.04618
[81]:https://www.linkedin.com/posts/akshay-pachaar_heres-a-common-misconception-about-rag-activity-7393265204421304320-T0jf?utm_source=share&utm_medium=member_desktop&rcm=ACoAABUa5xMBAWvx7L2IKhfsBuLjhTEWJhTYoNk
[82]:https://www.linkedin.com/posts/brijpandeyji_most-people-talk-about-rag-as-if-its-the-activity-7392913035993972736-hUgL/?utm_medium=ios_app&rcm=ACoAABUa5xMBAWvx7L2IKhfsBuLjhTEWJhTYoNk&utm_source=social_share_send&utm_campaign=mail
[83]:https://github.com/microsoft/agent-lightning
[84]:https://github.com/openpipe/ART
[85]:https://huggingface.co/spaces/mteb/leaderboard?benchmark_name=RTEB%28beta%29&utm_source=fnf&utm_medium=linkedin&utm_term=achawla&utm_content=embedding-leaderboard-rteb%28beta%29
[86]:https://github.com/toon-format/toon
[87]:https://github.com/CHATS-lab/verbalized-sampling
[88]:https://arxiv.org/pdf/2510.01171
[89]:[https://github.com/FalkorDB/FalkorDB](https://www.linkedin.com/posts/akshay-pachaar_microsoft-google-aws-everyones-trying-activity-7398702041004503041-kJgu?utm_source=share&utm_medium=member_desktop&rcm=ACoAABUa5xMBAWvx7L2IKhfsBuLjhTEWJhTYoNk)
[90]:https://www.kaggle.com/whitepaper-prototype-to-production
