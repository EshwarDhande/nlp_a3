# TO BE EDITED:
Individual Contributions just copied over from the previous assignment!
Placeholders:
- classification_model_parameters
- classification_model_link
- classification_model_accuracy
- classification_model_precision
- classification_model_recall
- classification_model_f1
- classification_train_epoch
- qa_model_parameters
- qa_model_link
- qa_model_exact_match
- qa_model_blue
- qa_model_rouge_1
- qa_model_rouge_2
- qa_model_rouge_l
- qa_model_meteor
- qa_model_f1
- qa_train_epoch
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

Parameter Calculation is shown <a href = 'https://github.com/EshwarDhande/nlp_a3/blob/master/sst.ipynb'>here.</a>

## **Question 3a, 4a, 5, 6, 7: Fine-tuning for Classification** 
- Used `transformers.AutoModelForSequenceClassification` to load the pre-trained model for fine-tuning on SST-2 dataset.
  
- The model loaded using the above method had classification_model_parameters parameters.

- Link to the huggingface model: [classification_model_link](https://huggingface.co/ishanarang/gemma2_finetune_sst2)
  
- The following metrics were calculated before training:
  
| Metric      | Score                               |
|-------------|-------------------------------------|
| Accuracy    | 0.551605504587156       |
| Precision   | 0.5591374091843514      |
| Recall      | 0.551605504587156         |
| F1          | 0.5428627070782336             |
  
- The following metrics were calculated after training for 3 epochs:
  
<img src = 'https://github.com/EshwarDhande/nlp_a3/blob/master/ft_isha.png'>

## **Question 3b, 4b, 5, 6, 7: Fine-tuning for Question-Answering** 
- Extended `torch.nn.Module` to create a class `ExtractiveQAModel`, adding a feed-forward layer to the base model.
  
- The model created using the above method had qa_model_parameters parameters.

- Link to the huggingface model: qa_model_link
  
- The following metrics were calculated after training for qa_train_epoch epochs:
  
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

**Isha and Manish** - Understanding sst2 dataset. Optimizing to run in kaggle file by various methods. Finetuning gemma on sst2 by utilizing peft technique.

**Eshwar, Mukul and Preyum** - Understanding squad dataset. Optimizing fine tuning technique to run on colab. Fine-tuning gemma on squad/


