from sklearn.model_selection import KFold
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from datasets import load_metric
from IPython.core.display import display

import numpy as np
import pandas as pd

# kind of metric
metric = load_metric('accuracy')
# calculate metric function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 5 fold cross validation training function
def cross_validation(train_and_valid_ds, test_ds, data_collator, args, training_args):
    test_accuracy = []
    kf = KFold(n_splits=5, shuffle=True, random_state=training_args.seed)
    num=0
    for train_idx, valid_idx in kf.split(train_and_valid_ds):
        num+=1
        model = AutoModelForSequenceClassification.from_pretrained(args.pretrain_path, num_labels=3)

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
            'pred': outputs.predictions.argmax(axis=-1),
            'label': outputs.label_ids
        })
        preds_df.to_csv(training_args.output_dir + '/result.csv')
        
        # make all fold validation table
        evaluation = trainer.evaluate(test_ds)
        test_accuracy.append([str(num)+'th', evaluation['eval_accuracy']*100])
    
    training_args.output_dir = training_args.output_dir.replace('/cross_validation_5', '')
    # add all fold validation table to average ± std
    columns = ['CV', 'test_acc']
    test_accuracy_np = np.array([x[1] for x in test_accuracy])
    test_accuracy.append(['final result', str(sum(test_accuracy_np)/len(test_accuracy_np))+'±'+str(np.std(test_accuracy_np))])
    df = pd.DataFrame(data=test_accuracy, columns=columns).set_index('CV')
    df.to_csv(training_args.output_dir + '/evaluation.csv')
    display(df)