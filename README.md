# TO BE EDITED:
Individual Contributions just copied over from the previous assignment!
Placeholders:
- classification_model_parameters
- classification_model_link
- classification_model_accuracy
- classification_model_precision
- classification_model_recall
- classification_model_f1
- qa_model_parameters
- qa_model_link
- qa_model_exact_match
- qa_model_blue
- qa_model_rouge_1
- qa_model_rouge_2
- qa_model_rouge_l
- qa_model_meteor
- qa_model_f1
---
# NLP_Assignment_3
---
The code for fine-tuning for classification can be found <a href = 'https://github.com/EshwarDhande/nlp_a3/blob/main/sst.ipynb'>here</a> and 
code for fine-tuning for question-answering can be found <a href = 'https://github.com/EshwarDhande/nlp_a3/blob/main/Squad.ipynb'>here</a>.
---
## Questions 1 and 2:
We picked the Gemma-2 2B model (`gemma-2-2b-it`) and calculated the parameters of the model by summing up the parameters of each layer of the model and compared against those reported in the technical paper for the model.
We note a small difference of 295,000. This discrepancy is minimal and could be due to rounding errors or slight variations in how the model was initialized or structured in the implementation.
This is a negligible difference, indicating that the loaded model closely follows the architecture described in the paper.

## **Question 3a, 4a, 5, 6, 7: Fine-tuning for Classification** 
- Used `transformers.AutoModelForSequenceClassification` to load the pre-trained model for fine-tuning on SST-2 dataset.
  
- The model loaded using the above method had classification_model_parameters parameters.

- Link to the huggingface model: classification_model_link
  
- The following metrics were calculated after training for train_epoch epochs:
  
| Metric      | Score                               |
|-------------|-------------------------------------|
| Accuracy    | classification_model_accuracy       |
| Precision   | classification_model_precision      |
| Recall      | classification_model_recall         |
| F1          | classification_model_f1             |

## **Question 3b, 4b, 5, 6, 7: Fine-tuning for Question-Answering** 
- Extended `torch.nn.Module` to create a class `ExtractiveQAModel`, adding a feed-forward layer to the base model.
  
- The model created using the above method had qa_model_parameters parameters.

- Link to the huggingface model: qa_model_link
  
- The following metrics were calculated after training for train_epoch epochs:
  
| Metric       | Score                   |
|--------------|-------------------------|
| Exact Match  | qa_model_exact_match    |
| BLEU         | qa_model_blue           |
| ROUGE-1      | qa_model_rouge_1        |
| ROUGE-2      | qa_model_rouge_2        |
| ROUGE-L      | qa_model_rouge_l        |
| METEOR       | qa_model_meteor         |
| F1           | qa_model_f1             |


## ***Individual Contribution***

**Manish, Eshwar, and Isha** - Curated five different samples and trained SentencePieceBPETokenizer, BertWordPieceTokenizer, ByteLevelBPETokenizer and went ahead with Sentence Piece because of the best results. Calculated fertility score for all the samples. Tokenized the dataset. 

**Mukul and Preyum** - Selected Llama model and reduced parameters. Trained reduces llama model with the trained tokenizer. Calculated training loss after every 100 steps and perplexity after every 0.1 epochs. 


