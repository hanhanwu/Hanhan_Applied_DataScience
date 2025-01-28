# AI Learning Notes

## Promising to Explore
### Encoders (for Embeddings)
* [ColBERT][26]
  * Enables more effective retrieval than bi-encoder, more efficient than cross-encoder
  * ColBERT uses the MaxSim operation, which computes the maximum similarity between query and document token embeddings. This allows for <b>fine-grained matching at the token level</b>, leading to better relevance estimation, especially <b>in cases where specific terms in the query are critical for retrieval</b>.

### RAG Alternatives
* [CAG (Cache-augmented generation), a faster alternative for RAG][19]

### RAG Techniques
* [RAG Techniques][18]
* Text Chunking
  * [Explain how does chunking work][22] 
  * [LangChain's Text Spliter][13] vs [Chonkie][14]
  * [Langchain has more options for embeddings, retrievers, etc.][16]

### Agents
* [Khoj - A Few Clicks to Create Your Own Agents][15]
* [Browser Use - Allow agents to interact with the browser][23]
* [Agentic AI Design Patterns][27]

### Evaluation
* [Trustworthy Language Model (TLM)][20], [reduce Hallucinations][21] for any LLM
  * [Use TLM in LangChain][25]

### LLM Model Fine Tuning
* [How to fine tune DeepSeek R1][30]
  * R1 is reasoning model 
  * [the completed code][29] 


## LLM Visualization
* [Visually to understand LLM models in detail][28] 💖


## LangChain Universe 😉
* [LangChain's RAG chatbot][24], much better than its tutorials, it allows to use different LLMs and can write code for you! 💖
* [LangSmith Cookbook][8], different ways to test & evaluate & optimization
  * LangServer is replaced by LangGraph
* [Prompts in LangChain Hub][12]

### LangGraph
* [LangGraph github][9]: build agent workflows
* [LangGraph Basics][17]: how to use Time Travel, Human in the Loop, etc.
* [LangGraph exercise][11] 💖


## LlamaIndex Examples
* [LlamaIndex Examples][2]
* [Emsemble Retriever][4]
  * LlamaIndex specific 📍
  * It can retrieve data from different sources 
* [Retrieval][1]
  * File retriever vs Chunk retriever
    * File-level retrieval is useful for handling user questions that require the entire document context to properly answer the question. It can be slower and more expensive than chunk level retrieval. 
  * Build an agent to decide file-level or chunk-level retrieving
  * Auto Retrieval: How to make file-level retrieval more sophisticated by allowing the LLM to infer a set of metadata filters, based on some relevant example documents
    * [How does auto retrieval work][5]
    * [Specific example of auto retrieval][6]
      * It generates inferred query, also generates inferred filters <b>based on real examples from the text</b>
      * If the user query is too shot, how does this inferred query look like? 🤔
    * 🌟 Inspiration: build an agent using auto retrieval at doc level and chunk level
* [Subquestion Query Engine][3]
  * Break down a complex query into sub queries
* [Corrective RAG Workflow][7]
  * It uses LLM to evaluate the retrieval relevancy, for non-relevant nodes (chunks), it uses LLM to adjust the query to improve its search performance and search the info online.
  * 🌟 Inspirations:
    * The LLM prompt to check retrieval relevancy
    * LLM to transform the query to improve search performance
    * Searching online can be an option when the retrieved node is irrelevant
  * Does this method guarantee the online searched resluts are more relevant than the "irrelevant node"? 🤔

 
## LlamaIndex vs Langchain
* [DataCamp Comparison][10]
  * LlamaIndex is good at indexing large datasets and retrieving relevant information quickly and accurately --> use it when information retrieval is critica
  * Langchain is modular, flexible to customize
  * 🌟 Inspirations:
    * How about use LlamaIndex for retrieval and use Langchain for others in LangGraph 🤔


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
