import torch
import torch.optim as optim
import time
import os
from sklearn.metrics import precision_score, recall_score, f1_score

from preprocessing import prepare_sequence, read_data
from model import BiLSTM_CRF
from sklearn.model_selection import train_test_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 30
HIDDEN_DIM = 100
EPOCHS = 1000
DROPOUT = 0.3
JSON_TRAIN = "dataset/train.json"
JSON_TEST = "dataset/test.json"
print("------------START PREPROCESSING-------------")
training_data = read_data(JSON_TRAIN)
test_data = read_data(JSON_TEST)
# Splitting document
# training_data, test_data = train_test_split(data, test_size=0.3, random_state=589)

best_f1 = 0
best_epoch = 0
word_to_ix = {"<PAD>": 0, "<UNK>": 1}
tag_to_ix = {}

# Build vocab of dataset
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    for tag in tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)

tag_to_ix["<START>"] = len(tag_to_ix)
tag_to_ix["<STOP>"] = len(tag_to_ix)
print("------------END PREPROCESSING-------------")

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, DROPOUT, device)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
t1 = time.time()
for epoch in range(
        EPOCHS):
    print("------------EPOCH", epoch, "-------------")
    model.train()
    all_predictions = []
    all_labels = []
    print("------------START TRAINING-------------")
    for sentence, tags in training_data:
        # We need to clear gradients out before each instance
        model.zero_grad()

        # Turn inputs into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix).to(device)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long).to(device)

        # Run the forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Compute the loss, gradients, and update the parameters
        loss.backward()
        optimizer.step()
    print("------------START EVALUATION-------------")

    # Validation after each epoch
    model.eval()  # Evaluation mode

    # Iterate to see the metrics on each epoch
    for sentence, tags in test_data:
        # Transform tags to index
        tag = [tag_to_ix[t] for t in tags]
        
        # Make the prediction for th sentences of validation
        sentence_in = prepare_sequence(sentence, word_to_ix).to(device)
        lstm_feats = model(sentence_in)

        # Get the prediction of the tags
        predictions = lstm_feats[1]
        all_predictions.extend(predictions)
        all_labels.extend(tag)

    # Calculate the metrics
    precision = precision_score(all_labels, all_predictions, average='micro')
    recall = recall_score(all_labels, all_predictions, average='micro')
    f1 = f1_score(all_labels, all_predictions, average='micro')

    # Show the metrics each 10 epochs
    print(f"-----------Epoch {epoch+1}/{EPOCHS}    F1-score {f1:.4f}    Recall {recall:.4f}    Precision {precision:.4f}-----------")

    # Save the best and last model
    if f1 > best_f1:
        best_f1 = f1
        best_recall = recall
        best_precision = precision
        best_epoch = epoch
        os.system("rm best*.pt")
        torch.save(model.state_dict(), 'best-model' + str(best_epoch) + '_' + str(best_f1) + '_' + str(best_recall) + '_' + str(best_precision) + '.pt')
        print(f"------New Best model: F1-score {best_f1:.4f}, Recall {best_recall:.4f}, Precidion {best_precision:.4f}, Epoch {best_epoch}-----")
    # Break the training if the score is not improving after 50 epochs
    if epoch - best_epoch > 25:
        break
    torch.save(model.state_dict(), 'last-model.pt')
t2 = time.time()
# Show the best model scores
print(f"Best model: F1-score {best_f1:.4f}, Recall {best_recall:.4f}, Precidion {best_precision:.4f}, Time {t2-t1}")

# Check predictions after training
# with torch.no_grad():
#     precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
#     print(precheck_sent)
#     print(model(precheck_sent))
#     print(model._viterbi_decode(model(precheck_sent)))
#     print(model(precheck_sent)[1])
