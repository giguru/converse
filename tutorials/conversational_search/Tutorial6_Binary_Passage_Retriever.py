from haystack import Pipeline
from haystack.retriever.anserini import DenseAnseriniRetriever

# LOAD COMPONENTS
retriever = DenseAnseriniRetriever(prebuilt_index_name="wikipedia-bpr-single-nq-hash",
                                   binary=True,
                                   query_encoder="castorini/bpr-nq-question-encoder")

# BUILD PIPELINE
p = Pipeline()
p.add_node(component=retriever, name="Retriever", inputs=["Query"])

# RUN A QUERY
output = p.run(query="When was Elon Musk born?")
print(output['documents'])

exit()