# Data Engineering Notes

## MongoDB Atlas
* It's the MongoDB cloud service, you can deploy fully managed MongoDB across AWS, Azure, or GCP and create flexible and scalable databases.
* [Official Setup Tutorial][1]
* [PyMongo functions][2]
* [MongoDB Reference - more functions & params][3]
* [My Code - Summarize Different Index in MongoDB][4]
  * Single field index
  * Compound index
  * Multikey index
  * Text index
  * Geospetial index
  * Adding partial index when creating an index
### When to use MongoDB
* You plan to integrate hundreds of different data sources, the document-based model of MongoDB will be a great fit as it will provide a single unified view of the data
* When you are expecting a lot of reads and write operations from your application but you do not care much about some of the data being lost in the server crash
* You can use it to store clickstream data and use it for the customer behavioral analysis


[1]:https://docs.atlas.mongodb.com/getting-started/
[2]:https://www.w3schools.com/python/python_mongodb_insert.asp
[3]:https://docs.mongodb.com/manual/reference/
[4]:https://github.com/hanhanwu/Hanhan_Applied_DataScience/blob/master/data_engineering/mongo_DB_index.ipynb
