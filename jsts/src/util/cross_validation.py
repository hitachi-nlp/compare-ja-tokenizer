import os
from sklearn.model_selection import KFold
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from datasets import load_metric
from IPython.core.display import display

import numpy as np
import pandas as pd

# kind of metric
metric = load_metric("glue", "stsb")
# calculate metric function
def compute_metrics(p):
    preds = p.predictions
    labels = p.label_ids
    preds = np.squeeze(preds)
    print(preds)
    print(labels)
    result = metric.compute(predictions=preds, references=labels)
    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()
    return result


# 5 fold cross validation training function
def cross_validation(train_and_valid_ds, test_ds, data_collator, args, training_args):
    test_peason = []
    test_spearman = []
    cv = []
    kf = KFold(n_splits=5, shuffle=True, random_state=training_args.seed)
    num=0
    for train_idx, valid_idx in kf.split(train_and_valid_ds):
        num+=1
        if args.pretokenizer_type == 'cl-tohoku/bert-base-japanese':
            model = AutoModelForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese', num_labels=1)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(args.pretrain_path, num_labels=1)
        # split datasets into fold
        train_fold = train_and_valid_ds.select(train_idx)
        valid_fold = train_and_valid_ds.select(valid_idx)
        
        # train
        training_args.output_dir = training_args.output_dir.rstrip(str(num-1))
        training_args.output_dir = training_args.output_dir + str(num)
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_fold,
            eval_dataset=valid_fold,
            compute_metrics=compute_metrics
        )
        trainer.train()
        trainer.save_model(training_args.output_dir + '/best_model.bin')

        # error analisys
        outputs = trainer.predict(valid_fold)
        preds_df = pd.DataFrame({
            'pred':  [k[0] for k in outputs.predictions],
            'label': outputs.label_ids
        })
        preds_df.to_csv(training_args.output_dir + '/result.csv')
        
        # make all fold validation table
        evaluation = trainer.evaluate(test_ds)
        test_peason.append([str(num)+'th', evaluation['eval_pearson']*100])
        test_spearman.append([str(num)+'th', evaluation['eval_spearmanr']*100])
    
    columns_p = ['CV', 'Pearson']
    columns_s = ['CV', 'Spearman']
    test_peason_np = np.array([x[1] for x in test_peason])
    print("----------------")
    cv.append('final result')
    print(cv)
    test_peason.append(['final result', str(sum(test_peason_np)/len(test_peason_np))+'±'+str(np.std(test_peason_np))])
    test_spearman_np = np.array([x[1] for x in test_spearman])
    test_spearman.append(['final result', str(sum(test_spearman_np)/len(test_spearman_np))+'±'+str(np.std(test_spearman_np))])
    print(cv)
    print(test_peason)
    print(test_spearman)
    training_args.output_dir = training_args.output_dir.rstrip('/cross_validation_' + str(num))
    os.makedirs(training_args.output_dir+'/pred_result', exist_ok=True)
    df = pd.DataFrame(data=test_peason, columns=columns_p).set_index('CV')
    df.to_csv(training_args.output_dir+'/pred_result/evaluation_ex.csv')
    display(df)
    df = pd.DataFrame(data=test_spearman, columns=columns_s).set_index('CV')
    df.to_csv(training_args.output_dir+'/pred_result/evaluation_f.csv')
    display(df)
