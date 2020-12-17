import re
import os

import torch
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertModel, BertForPreTraining
from transformers import BertTokenizer, BertModel

aa = torch.tensor([[1,2], [3,4]], dtype=torch.long)
from transformers.convert_bert_original_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch

device='cuda' if torch.cuda.is_available() else 'cpu'

# from transformers.convert_bert_original_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch
# pytorch_model = convert_tf_checkpoint_to_pytorch("/Data/public/Bert/chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt",
#                                                  "/Data/public/Bert/chinese_wwm_ext_L-12_H-768_A-12/bert_config.json",
#                                                  "/Data/public/Bert/chinese_wwm_ext_L-12_H-768_A-12/pytorch_bert_model.bin")
model = torch.load("/Data/public/Bert/chinese_wwm_ext_L-12_H-768_A-12/pytorch_bert_model.bin")
from transformers import PreTrainedModel

# Loading from a TF checkpoint file instead of a PyTorch model (slower)
# config = BertConfig.from_json_file('/Data/public/Bert/chinese_wwm_ext_L-12_H-768_A-12/bert_config.json')
# model = BertForPreTraining.from_pretrained('/Data/public/Bert/chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt.index', from_tf=True, config=config)
# model.save_pretrained("/Data/public/Bert/chinese_wwm_ext_L-12_H-768_A-12/hg_tf/")

# model = TFBertModel.from_pretrained("/Data/public/Bert/chinese_wwm_ext_L-12_H-768_A-12/pytorch_bert_model.bin", from_pt=True, config=)
# PreTrainedModel.save_pretrained(model,)
tokenizer = BertTokenizer.from_pretrained("/Data/public/Bert/chinese_wwm_ext_L-12_H-768_A-12/vocab.txt")
model = BertForSequenceClassification.from_pretrained(
    "/Data/public/Bert/chinese_wwm_ext_L-12_H-768_A-12/hg_tf/", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False# Whether the model returns all hidden-states.
)
def rm_tags(text):
    re_tags = re.compile(r'<[^>]+>')
    return re_tags.sub(' ', text)


def read_files(filetype):
    path = "aclImdb/"
    file_list = []

    positive_path = path + filetype + "/pos/"
    for f in os.listdir(positive_path):
        file_list += [positive_path + f]

    negative_path = path + filetype + "/neg/"
    for f in os.listdir(negative_path):
        file_list += [negative_path + f]

    print("read", filetype, "files:", len(file_list))

    all_labels = ([1] * 12500 + [0] * 12500)

    all_texts = []
    for fi in file_list:
        with open(fi, encoding='utf8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]

    return all_labels, all_texts


y_train, train_text = read_files("train")
y_test, test_text = read_files("test")



# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
# input_ids = tokenizer.encode("i love you !",add_special_tokens=True,max_length=128)
# Loading from a TF checkpoint file instead of a PyTorch model (slower)
# config = BertConfig.from_json_file('/Data/public/Bert/chinese_wwm_ext_L-12_H-768_A-12/bert_config.json')
# model = BertModel.from_pretrained('/Data/public/Bert/chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt.index', from_tf=True, config=config)

sentences=train_text
labels=y_train

test_sentences=test_text
test_labels=y_test

MAX_LEN=128
# sentences = sentences[:,1:10]

input_ids = [tokenizer.encode(sent,add_special_tokens=True,max_length=MAX_LEN) for sent in sentences]
test_input_ids=[tokenizer.encode(sent,add_special_tokens=True,max_length=MAX_LEN) for sent in test_sentences]

from keras.preprocessing.sequence import pad_sequences
print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",
                          value=0, truncating="post", padding="post")

test_input_ids = pad_sequences(test_input_ids, maxlen=MAX_LEN, dtype="long",
                          value=0, truncating="post", padding="post")

# Create attention masks
attention_masks = []

# For each sentence...
for sent in input_ids:
    # Create the attention mask.
    #   - If a token ID is 0, then it's padding, set the mask to 0.
    #   - If a token ID is > 0, then it's a real token, set the mask to 1.
    att_mask = [int(token_id > 0) for token_id in sent]

    # Store the attention mask for this sentence.
    attention_masks.append(att_mask)

test_attention_masks = []

# For each sentence...
for sent in test_input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    test_attention_masks.append(att_mask)


from sklearn.model_selection import train_test_split

# Use 90% for training and 10% for validation.
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                            random_state=2020, test_size=0.1)
# Do the same for the masks.
train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,
                                             random_state=2020, test_size=0.1)




train_inputs = torch.Tensor(train_inputs)
validation_inputs = torch.Tensor(validation_inputs)
test_inputs=torch.Tensor(test_input_ids)

train_labels = torch.Tensor(train_labels)
validation_labels = torch.Tensor(validation_labels)
test_labels=torch.Tensor(test_labels)

train_masks = torch.Tensor(train_masks)
validation_masks = torch.Tensor(validation_masks)
test_masks=torch.Tensor(test_attention_masks)

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# The DataLoader needs to know our batch size for training, so we specify it
# here.
# For fine-tuning BERT on a specific task, the authors recommend a batch size of
# 16 or 32.

batch_size = 16

# Create the DataLoader for our training set.
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# Create the DataLoader for our test set.
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)


# Load BertForSequenceClassification, the pretrained BERT model with a single
# linear classification layer on top.
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, force_download=True# Whether the model returns all hidden-states.
)

# Tell pytorch to run this model on the GPU.
model.cuda()



optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

from transformers import get_linear_schedule_with_warmup

# Number of training epochs (authors recommend between 2 and 4)
epochs = 2

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

import numpy as np


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


import time
import datetime


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


import random

seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Store the average loss after each epoch so we can plot them.
loss_values = []

# For each epoch...
for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):

        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)

            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)

        # The call to `model` always returns a tuple, so we need to pull the
        # loss value out of the tuple.
        loss = outputs[0]
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy

        # Track the number of batches
        nb_eval_steps += 1

    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

print("")
print("Training complete!")

print("dd")