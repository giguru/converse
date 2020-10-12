Retriever query embedding model was _facebook/dpr-question_encoder-single-nq-base_.

Retriever passage embeeding model was _facebook/dpr-ctx_encoder-single-nq-base_.

Reader was _deepset/roberta-base-squad2_.
```
10/12/2020 15:44:38 - INFO - converse.src.converse -   518.0 out of 3427 questions were correctly answered 15.12%).
10/12/2020 15:44:38 - INFO - converse.src.converse -   2544.0 questions could not be answered due to the retriever.
10/12/2020 15:44:38 - INFO - converse.src.converse -   365.0 questions could not be answered due to the reader.

___Retriever Metrics in Finder___
Retriever Recall            : 0.258
Retriever Mean Avg Precision: 0.258

___Reader Metrics in Finder___
Top-k accuracy
Reader Top-1 accuracy             : 0.314
Reader Top-1 accuracy (has answer): 0.269
Reader Top-k accuracy             : 0.587
Reader Top-k accuracy (has answer): 0.526
Exact Match
Reader Top-1 EM                   : 0.087
Reader Top-1 EM (has answer)      : 0.009
Reader Top-k EM                   : 0.151
Reader Top-k EM (has answer)      : 0.026
F1 score
Reader Top-1 F1                   : 0.212
Reader Top-1 F1 (has answer)      : 0.152
Reader Top-k F1                   : 0.389
Reader Top-k F1 (has answer)      : 0.300
No Answer
Reader Top-1 no-answer accuracy   : 0.619
Reader Top-k no-answer accuracy   : 1.000

___Time Measurements___
Total retrieve time           : 656.618
Avg retrieve time per question: 0.192
Total reader timer            : 230.222
Avg read time per question    : 0.260
Total Finder time             : 887.025
```