# Mind-Map-Generator

We developed a Mind Map Generator, which summarizes given text to make it simpler and quicker to interpret. <br>
To generate a mind map, the user gives input in three ways: raw text, URL, and file. If the user provides a URL as input, we use web scrapping to get the text, or if the user uploads a file as input, we will load the file and extract the text. The model will take this text and clean the input the user provides. We will then use the Text Rank Algorithm to summarize the given text. Once we summarize the text, the model will run an LDA topic modeling algorithm to extract the topic and relevant sentences from the summarized text. Based on the extracted topics, we will use the RAKE algorithm to get relevant keywords to each topic from the sentences. Now, after getting the required words from these, we will take only the top 75th percentile of the words based on their probability score of the rake algorithm, which gave us full keywords. After this, we added a similarity check between the keywords using cosine similarity to remove similar keywords from the list. Finally, based on these keywords, we will generate the mind map. <br>
We have also developed a web application using Flask to improve the user interaction. The webpage where the user gives the input is as follows:
![image](https://github.com/NotManigandan/Mind-Map-Generator/assets/72668299/5a7b9e3a-36d0-47fe-87a6-ef4bb45b09d5)
<br>
The webpage where the mind map is displayed is as follows:
![Screenshot 2024-01-07 at 19-24-16 Flask Summarizer - Result](https://github.com/NotManigandan/Mind-Map-Generator/assets/72668299/6abf4401-23db-4022-8c88-beb6ea34af30)
