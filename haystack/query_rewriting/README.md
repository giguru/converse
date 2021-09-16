# Conversational Query Reformulators

This Framework offers several options for query reformulators. Query Reformulators take a query and the conversation
history and combines them into a single/adjusted/extended string. 

**Example conversation**
- Q: Who founded Microsoft?
- A: Bill Gates.
- Q: Is he a smart guy?
- A: Yes, he is.
- Q: Where does he live?

Model | Results
--- | ---
ConcatenationReformulator() | Who founded Microsoft? Bill Gates. Is he a smart guy?  Yes, he is. Where does he live?
PrependingReformulator(history_window=2) | Is he a smart guy? Yes, he is. Where does he live?
ClassificationReformulator | Where does he live? Bill Gates
GenerativeReformulator | Where does Bill Gates live?
CustomFilterReformulator | ??? (It depends on the filter function you provide to it.)

## Heuristic/Deterministic Reformulator

### ConcatenationReformulator
It simple concatenates all the history and the query.

### PrependingReformulator
It prepends a certain history window to the current query. It also has the option to always append the first history
item.

### CustomFilterReformulator
The `__init__` function takes a `filter_func: Callable` argument, which is the function it will use to reformulate the
query. 

## Transformer-based Reformulators

### GenerativeReformulator
It rewrites the current query to be context-independent. It is compatible with any Seq2Seq Language model.

### ClassificationReformulator 
It select the relevant terms from the conversation history and appends them to the current
query. It was created to be compatible with the [uva-irlab/quretec](https://huggingface.co/uva-irlab/quretec) model
hosted on Huggingface.
 
