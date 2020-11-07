# Data Engineering Notes

## Relational Database
* [Super key, Candidate key, Primary key, Alternative key, Composite key, Foreign key][6]
* [SQl vs NoSQL][8]

## Hadoop Ecostsems
* [Brief intro about Hadoop ecosystem][9]
  * I like the "big data processing stages" diagrama
  * [Hive][12] - It can convert query into map reduce work which saves lots of effort to write map reduce code. But it doesn't work well with real time data.
* [HDFS Architecture][11]

## NoSQL General
### [NoSQL Categories][7]
* Document based
  * JSON format
  * Such as MongoDB, Orient DB, and BaseX
* Key-value based
  * Such as DynamoDB, Redis, and Aerospike
* Wide column based
  * Dynamic columns
  * Such as Cassandra and HBase
* Graph based
  * Neo4j, Amazon Neptune
* When to use Cassandra: Availability > consistency, more writing than reading, less join
* When to use ElasticSearch: fuzzy text search, log analysis
* When to use DynamoDB: Highly consistency required, large data size with simple key-values
* When to use HBase: PB size data, random & real time storage access


## [Consistency, Availability, Partition Tolerance (CAP) for distributed DB][5]
* "Consistency" - different nodes connection
  * It means that the user should be able to see the same data no matter which node they connect to on the system.
* "Availability" - client gets the response
  * It means that every request from the user should elicit a response from the system. Whether the user wants to read or write, the user should get a response even if the operation was unsuccessful.
* "Partition Tolerance" - break tolerance
  * It means a break in communication then Partition tolerance would mean that the system should still be able to work even if there is a partition in the system. Meaning if a node fails to communicate, then one of the replicas of the node should be able to retrieve the data required by the user.


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
* Mainly for Consistency & Partition Tolerance
* You plan to integrate hundreds of different data sources, the document-based model of MongoDB will be a great fit as it will provide a single unified view of the data
* When you are expecting a lot of reads and write operations from your application but you do not care much about some of the data being lost in the server crash
* You can use it to store clickstream data and use it for the customer behavioral analysis

## Spark
* [Tips to make spark job more efficient][10]
  * Configuring number of Executors, Cores, and Memory
  * Avoid Long Lineage
  * Broadcasting 
  * Larger partition of dataset --> to allow more parallelism on your job
  * Columnar File Formats, such as parquet
  * Use dataframe as much as possible, rather than RDD. Because spark daatfarmes had associated metadata to allow spark optimize the query plans
* [RDD vs Dataframe vs Dataset][15]
  * Dataset has more features but dataframe still has best performance
  
## Cloud Platforms
### Google BigQuery
* [console.cloud.google.com][13]
  * Search for "BigQuery"
* It also has a dashboard with it, feeling like redash (for redshift)
  * [A brief guide][14]
* Simple SQL


[1]:https://docs.atlas.mongodb.com/getting-started/
[2]:https://www.w3schools.com/python/python_mongodb_insert.asp
[3]:https://docs.mongodb.com/manual/reference/
[4]:https://github.com/hanhanwu/Hanhan_Applied_DataScience/blob/master/data_engineering/mongo_DB_index.ipynb
[5]:https://www.analyticsvidhya.com/blog/2020/08/a-beginners-guide-to-cap-theorem-for-data-engineering/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[6]:https://www.analyticsvidhya.com/blog/2020/07/difference-between-sql-keys-primary-key-super-key-candidate-key-foreign-key/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[7]:https://www.analyticsvidhya.com/blog/2020/09/different-nosql-databases-every-data-scientist-must-know/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[8]:https://www.analyticsvidhya.com/blog/2020/10/sql-vs-nosql-databases-a-key-concept-every-data-engineer-should-know/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[9]:https://www.analyticsvidhya.com/blog/2020/10/introduction-hadoop-ecosystem/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[10]:https://www.analyticsvidhya.com/blog/2020/10/how-can-you-optimize-your-spark-jobs-and-attain-efficiency-tips-and-tricks/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[11]:https://www.analyticsvidhya.com/blog/2020/10/hadoop-distributed-file-system-hdfs-architecture-a-guide-to-hdfs-for-every-data-engineer/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[12]:https://www.analyticsvidhya.com/blog/2020/10/getting-started-with-apache-hive/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[13]:console.cloud.google.com
[14]:https://www.analyticsvidhya.com/blog/2020/11/basic-introduction-to-google-bigquery-and-data-studio-every-data-scientist-should-know/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[15]:https://www.analyticsvidhya.com/blog/2020/11/what-is-the-difference-between-rdds-dataframes-and-datasets/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
