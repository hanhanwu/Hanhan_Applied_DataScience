# AI Learning Notes

## LlamaIndex Examples
* [LlamaIndex Examples][2]
* [Emsemble Retriever][4]
  * LlamaIndex specific 🦙
  * It can retrieve data from different sources 
* [Retrieval][1]
  * File retriever vs Chunk retriever
  * Build an agent to decide file-level or chunk-level retrieving
  * Auto Retrieval: How to make file-level retrieval more sophisticated by allowing the LLM to infer a set of metadata filters, based on some relevant example documents
    * [How does auto retrieval work][5]
    * [Specific example of auto retrieval][6]
      * It generates inferred query, also generates inferred filters <b>based on real examples from the text</b>
      * If the user query is too shot, how does this inferred query look like? 🤔
    * 🌟 Inspiration: build an agent using auto retrieval at doc level and chunk level
* [Subquestion Query Engine][3]
  * Break down a complex query into sub queries


[1]:https://github.com/run-llama/llamacloud-demo/blob/main/examples/10k_apple_tesla/demo_file_retrieval.ipynb
[2]:https://github.com/run-llama/llamacloud-demo/tree/main/examples
[3]:https://github.com/run-llama/llamacloud-demo/blob/main/examples/10k_apple_tesla/demo_subquestion.ipynb
[4]:https://github.com/run-llama/llamacloud-demo/blob/main/examples/10k_apple_tesla/demo_ensemble_retrieval.ipynb
[5]:https://github.com/run-llama/llamacloud-demo/blob/main/examples/advanced_rag/auto_retrieval_img.png
[6]:https://github.com/run-llama/llamacloud-demo/blob/main/examples/advanced_rag/auto_retrieval.ipynb
