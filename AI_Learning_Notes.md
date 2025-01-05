# AI Learning Notes

## Promising to Explore
### RAG Techniques
* [RAG Techniques][18]
* Text Chunking
  * [LangChain's Text Spliter][13] vs [Chonkie][14]
  * [Langchain has more options for embeddings, retrievers, etc.][16]

### Agents
* [Khoj - A Few Clicks to Create Your Own Agents][15]


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

 
## LangChain Universe 😉
* [LangSmith Cookbook][8], different ways to test & evaluate & optimization
  * LangServe is replaced by LangGraph
* [Prompts in LangChain Hub][12]

### LangGraph
* [LangGraph github][9]: build agent workflows
* [LangGraph Basics][17]: how to use Time Travel, Human in the Loop, etc.
* [LangGraph exercise][11] 💖


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
