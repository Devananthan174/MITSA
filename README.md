# MITSA
**EXPLORING MULTILINGUAL INDIAN TWITTER SENTIMENT ANALYSIS: A COMPARATIVE STUDY.**

Sentiment analysis is a valuable method for analyzing texts and understanding the opinions, attitudes, and emotions expressed toward different subjects. Its application to large-scale data, particularly on social media platforms like Twitter or Facebook, offers valuable insights. However, Twitter data presents unique challenges due to its complex and noisy nature, as well as its intricate syntactic and semantic structures. Moreover, analyzing sentiment becomes even more difficult when dealing with multimodal Twitter data in Indian languages. In order to address these obstacles, it is essential to create a robust framework capable of effectively processing Twitter content across diverse languages and encodings. To ascertain the most optimal methodology, we carry out extensive assessments employing multiple variations of multilingual and single-language models, ultimately relying on the evaluation outcomes to guide our selection process. Our aim is to create a comprehensive model capable of effectively characterizing all aspects of Twitter data. Through a comprehensive examination of the data, we gain a deeper understanding of the sentiments expressed across various languages used on Twitter in India. These results offer valuable insights into the diverse range of emotions, opinions, and attitudes expressed by Indian Twitter users, enabling a more nuanced understanding of public sentiment in this multilingual context.

![image](https://github.com/Devananthan174/MITSA/assets/94842136/d90cb188-228a-4e68-abd9-70ba1fac0d37)


**Proposed Methodology:**

1. Utilizing TWINT for data extraction from Twitter: TWINT is a Python library that allows for data extraction from Twitter without the need for an API. The methodology involves using TWINT to collect relevant tweets for sentiment analysis. This step ensures a diverse and real-time dataset for analysis.

2. Performing preprocessing techniques: The extracted Twitter data needs to be preprocessed to convert it into a structured format suitable for sentiment analysis. Preprocessing techniques may include cleaning the data by removing URLs, special characters, and irrelevant information. It may also involve tokenization, normalization, and handling of emojis or emoticons.

3. Applying various models and selecting the most suitable one: The preprocessed dataset is then used to train and evaluate different sentiment analysis models. The proposed models available for consideration are "bert-base-multilingual-cased", "xlm-roberta-large," and "distilbert-base-multilingual-cased". These models are state-of-the-art multilingual language models commonly used for sentiment analysis tasks. The models' performance, accuracy, and efficiency are evaluated, and the most suitable model is chosen based on the results.

4. Presenting the results: The results obtained from the selected model are presented, which may include metrics such as accuracy, precision, recall, or F1 score. The analysis of the sentiment distribution across languages and any notable patterns or insights discovered in the dataset are also included in the presentation of results.

By following this proposed methodology, the study aims to leverage TWINT for Twitter data extraction, preprocess the data, apply multiple models for sentiment analysis, and select the most suitable model based on performance. The final step involves presenting the results, which provide insights into sentiment analysis in multilingual contexts using the chosen model.

![image](https://github.com/Devananthan174/MITSA/assets/94842136/7e506a92-f07a-4c6a-9ecf-9d9950942901)

**Dataset Description: DITSA_plus (our unique contribution).**

The DITSA_plus dataset is a collection of manually labeled data used for sentiment analysis. It contains 499,826 samples or instances of text, with each sample assigned one of three sentiment labels: Positive, Negative, or Neutral. The dataset is designed to support sentiment analysis tasks across multiple languages, including English, Hindi, Malayalam, Tamil, Telugu, and Kannada. By providing a diverse range of languages and sentiment classes, the dataset allows researchers and practitioners to develop and evaluate sentiment analysis models that can handle different languages and capture a variety of sentiments. The availability of such a large and multilingual dataset enables comprehensive analysis and modeling of sentiment across multiple languages.

**Experimental Setup and Results:**

The sentiment analysis task involves three classes or labels: Positive, Negative, and Neutral. The model is trained to classify text samples into one of these three sentiment categories. This multi-class classification problem allows for a comprehensive analysis of sentiment expressed in the input text. The training process utilizes a learning rate of 2.4245311885086787e-05. A batch size of 64 is employed during training. The optimizer selected for training is AdamW, which is an extension of the Adam optimizer. AdamW incorporates weight decay regularization, penalizing large weights in the model and preventing overfitting. By employing AdamW, the model can optimize its parameters effectively while controlling model complexity. The model is trained for a total of 10 epochs. A multiclass confusion matrix is a concise and informative representation of the performance of a multiclass classification model. It consists of rows and columns, where each row represents the true class labels and each column represents the predicted class labels. The elements within the matrix provide a count or proportion of instances that belong to a particular true class and were predicted as a specific class. By analyzing the confusion matrix, valuable insights can be gained about the model's accuracy, precision, recall, and F1-score for each class, as well as the patterns of correct and incorrect predictions, enabling a comprehensive evaluation of the model's predictive capabilities across multiple classes. To evaluate the performance of the model, the primary metric used is the confusion matrix. In terms of hardware, the experiments are conducted using an NVIDIA A100-SXM4-40GB GPU card. 

To enhance the performance of the multilingual models, we incorporated four task-specific layers: Dropout, 1D convolutional, bidirectional LSTM, and a linear layer. After implementing fine-tuning on the dataset, we evaluated the performance of each model and selected the best-performing one based on the evaluation results.

![image](https://github.com/Devananthan174/MITSA/assets/94842136/a2e34080-1222-44c9-90af-cd0efbc0c994)



**Conclusion:**

This innovative approach represents the first utilization of multilingual models in Indian languages. It signifies a significant advancement in leveraging these models to analyze sentiments across diverse Indian language contexts, addressing the challenges of language diversity and limited resources. By adopting this pioneering method, we can enhance sentiment analysis capabilities and gain valuable insights from multiple Indian languages simultaneously. 

Among the three models evaluated in the experiment, the 'distilbert-base-multilingual-cased' model emerges as the most superior for sentiment analysis. This finding highlights its effectiveness in handling multilingual sentiment classification tasks even when computational resources are limited. The model's ability to achieve competitive performance with fewer parameters demonstrates its efficiency and potential practical application in real-world scenarios.

The results underscore the importance of developing efficient methods for handling large multilingual models while minimizing memory usage. As the field of natural language processing progresses, it becomes increasingly crucial to address the challenges of model compression and resource utilization. Finding effective techniques to optimize model size, performance, and resource consumption is a promising direction for future research.

Moreover, future work should focus on exploring innovative approaches to fine-tuning and optimizing the 'distilbert-base-multilingual-cased' model specifically. Fine-tuning methods that enhance the model's performance without sacrificing its efficiency and resource requirements could further improve its applicability in various sentiment analysis tasks.

Additionally, efforts should be made to expand the dataset and include more diverse languages to improve the model's language coverage and generalizability. Incorporating additional languages into the training process would allow the model to capture a broader range of linguistic variations and nuances, enabling more accurate sentiment analysis across various language contexts.

