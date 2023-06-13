from sklearn.model_selection import KFold
from transformers import BertForTokenClassification
from transformers import Trainer
from datasets import load_metric
from IPython.core.display import display

import numpy as np
import pandas as pd
from .make_data import create_dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
from tqdm import tqdm

# kind of metric
metric = load_metric('accuracy')
# calculate metric function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# model
class BertForTokenClassification_pl(pl.LightningModule):        
    def __init__(self, model_name, num_labels, lr, tokenizer=None, texts_list=None, entities_list=None):
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer', 'texts_list', 'entities_list'])
        self.bert_tc = BertForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.tokenizer = tokenizer
        self.texts_list = texts_list
        self.entities_list = entities_list
        
    def training_step(self, batch, batch_idx):
        output = self.bert_tc(**batch)
        loss = output.loss
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        output = self.bert_tc(**batch)
        val_loss = output.loss
        self.log('val_loss', val_loss)
        """
        entities_predicted_list = []
        for text in self.texts_list:
            entities_predicted = predict(text, self.tokenizer, self.bert_tc)
            entities_predicted_list.append(entities_predicted)
        result = evaluate_model(self.entities_list, entities_predicted_list)
        self.log('f_value', result['f_value'])
        """
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# 5 fold cross validation training function
def cross_validation(tokenizer, train_and_valid_ds,  args, training_args):
    # prepare datasets for training
    kf = KFold(n_splits=5, shuffle=True, random_state=training_args.seed)
    train_and_valid_idxs = kf.split(train_and_valid_ds)
    train_folds, valid_folds = [], []
    for train_idx, valid_idx in train_and_valid_idxs:
        train_fold = [train_and_valid_ds[idx] for idx in train_idx]
        valid_fold = [train_and_valid_ds[idx] for idx in valid_idx]
        dataset_train_for_loader, _, _ = create_dataset(
            tokenizer, train_fold, args.max_length
        )
        dataset_val_for_loader, valid_texts_list, valid_entities_list = create_dataset(
            tokenizer, valid_fold, args.max_length
        )
        # make dataloader for each fold
        dataloader_train = DataLoader(dataset_train_for_loader, batch_size=training_args.per_device_train_batch_size, shuffle=True)
        dataloader_valid = DataLoader(dataset_val_for_loader, batch_size=training_args.per_device_eval_batch_size)
        train_folds.append(dataloader_train)
        valid_folds.append(dataloader_valid)
    
    # train
    k=1
    ckpt_path_list = []
    for dataloader_train_fold, dataloader_valid_fold in zip(train_folds, valid_folds):
        checkpoint = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            save_weights_only=True,
            dirpath=training_args.output_dir + '/cross_validation_' + str(k),
            filename='model-{epoch}'
        )

        trainer = pl.Trainer(
            accelerator='cuda',
            devices=1,
            max_epochs=int(training_args.num_train_epochs),
            callbacks=[checkpoint],
            default_root_dir=training_args.output_dir
        )

        num_entity = 8
        model = BertForTokenClassification_pl(
            args.pretrain_path,
            num_labels=2*num_entity+1,
            lr=training_args.learning_rate,
            tokenizer=tokenizer, 
            texts_list=valid_texts_list,
            entities_list=valid_entities_list
        )

        trainer.fit(model, dataloader_train_fold, dataloader_valid_fold)
        ckpt_path_list.append(checkpoint.best_model_path)
        k+=1
    return ckpt_path_list


def predict(text, tokenizer, model):
    """
    Named Entity Recognition by BERT
    """
    # encode
    encoding, spans = tokenizer.encode_plus_untagged(
        text, return_tensors='pt'
    )
    encoding = { k: v.cuda() for k, v in encoding.items() }

    # calculate logits
    with torch.no_grad():
        output = model(**encoding)
        scores = output.logits
        labels_predicted = scores[0].cpu().numpy().tolist() 

    # convert labels to named entities
    entities = tokenizer.convert_bert_output_to_entities(
        text, labels_predicted, spans
    )

    return entities


def evaluate_model(entities_list, entities_predicted_list, type_id=None):
    """
    type_id: evaluate mdoels for all named entities if None, for the named entity if int.
    """
    num_entities = 0 
    num_predictions = 0 
    num_correct = 0 

    for entities, entities_predicted \
        in zip(entities_list, entities_predicted_list):

        if type_id:
            entities = [ e for e in entities if e['type_id'] == type_id ]
            entities_predicted = [ 
                e for e in entities_predicted if e['type_id'] == type_id
            ]
            
        get_span_type = lambda e: (e['span'][0], e['span'][1], e['type_id'])
        set_entities = set( get_span_type(e) for e in entities )
        set_entities_predicted = \
            set( get_span_type(e) for e in entities_predicted )

        num_entities += len(entities)
        num_predictions += len(entities_predicted)
        num_correct += len( set_entities & set_entities_predicted )

    precision = num_correct/(num_predictions+0.000001)
    recall = num_correct/(num_entities+0.000001)
    f_value = 2*precision*recall/(precision+recall+0.000001)

    result = {
        'num_entities': num_entities,
        'num_predictions': num_predictions,
        'num_correct': num_correct,
        'precision': precision,
        'recall': recall,
        'f_value': f_value
    }

    return result


def evaluation(args, training_args, ckpt_path_list, tokenizer, test_ds):
    eval_list = []
    num_entity = 8
    for ckpt_path in ckpt_path_list:
        model = BertForTokenClassification_pl(
            args.pretrain_path,
            num_labels=2*num_entity+1,
            lr=training_args.learning_rate,
            tokenizer=tokenizer,
            texts_list=[],
            entities_list=[]
            ).load_from_checkpoint(ckpt_path)
        bert_tc = model.bert_tc.cuda()

        entities_list = []
        entities_predicted_list = []
        for sample in tqdm(test_ds):
            text = sample['text']
            entities_predicted = predict(text, tokenizer, bert_tc)
            entities_list.append(sample['entities'])
            entities_predicted_list.append( entities_predicted )

        print("# answer")
        print(entities_list[0])
        print("# prediction")
        print(entities_predicted_list[0])

        eval_k = evaluate_model(entities_list, entities_predicted_list)
        eval_list.append(eval_k)

    columns = ['CV', 'precision', 'recall', 'f_value']
    eval_table = [['cv1'], ['cv2'], ['cv3'], ['cv4'], ['cv5'], ['ave']]
    for i in range(len(eval_list)):
        eval_table[i].append(eval_list[i]['precision'])    
        eval_table[i].append(eval_list[i]['recall'])    
        eval_table[i].append(eval_list[i]['f_value'])

    precisions = [eval_list[i]['precision'] for i in range(len(eval_list))]
    precisions = np.array(precisions)
    eval_table[-1].append(str(np.mean(precisions))+'±'+str(np.std(precisions)))

    recalls = [eval_list[i]['recall'] for i in range(len(eval_list))]
    recalls = np.array(recalls)
    eval_table[-1].append(str(np.mean(recalls))+'±'+str(np.std(recalls)))

    f_values = [eval_list[i]['f_value'] for i in range(len(eval_list))]
    f_values = np.array(f_values)
    eval_table[-1].append(str(np.mean(f_values))+'±'+str(np.std(f_values)))

    eval_df = pd.DataFrame(data=eval_table, columns=columns).set_index('CV')
    eval_df.to_csv(training_args.output_dir + '/evaluation_result.csv')
    print(eval_df)
    return
