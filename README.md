# Mind-Map-Generator

We developed Mind Map Generator which summarizes given text to make it simpler and quicker to interpret.
To generate a mind map, the user an give input through three ways: raw test, URL, file. If the user givesThe model will take this text and it will be cleaning the input provided by the user. We will be then using Text Rank Algorithm to summarize the given text. Once we summarize the text, the model will be running LDA topic modelling algorithm to extract the topic and relevant sentences to the topic from the summarized text. Based on the extracted topics, we will be using RAKE algorithm to get keywords relevant keywords to each topic from the sentences. Now after getting the required words from these, what we will be doing is to take only the top 75th percentile of the words based on their probability score of the rake algorithm which gave us full keywords. After this we added a similarity check between the keywords using cosine similarity so that similarly keywords are removed from the list. Finally, based on these keywords, we will be generating the mind map.


![image](https://github.com/NotManigandan/Mind-Map-Generator/assets/72668299/5a7b9e3a-36d0-47fe-87a6-ef4bb45b09d5)


![Screenshot 2024-01-07 at 19-24-16 Flask Summarizer - Result](https://github.com/NotManigandan/Mind-Map-Generator/assets/72668299/6abf4401-23db-4022-8c88-beb6ea34af30)
