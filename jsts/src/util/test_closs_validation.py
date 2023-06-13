from sklearn.model_selection import KFold
from transformers import AutoModelForQuestionAnswering
from transformers import Trainer
from datasets import load_metric
from IPython.core.display import display
import numpy as np
import pandas as pd
import collections
from .makedata import TokenizerFuncClass

#formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
#references = [{"id": ex["id"], "answers": ex["answers"]} for ex in test_ds]
#res = metric.compute(predictions=formatted_predictions, references=references)
#test_accuracy.append([str(num)+'th', res["exact_match"] / 100])
#test_f1.append([str(num)+'th', res["f1"] / 100])


# 5 fold cross validation training function
def test_cross_validation(tokenize_func, train_and_valid_ds, test_ds, data_collator, args, training_args):
    
    # calculate metric function
    def compute_metrics(eval_pred):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
        inputs = eval_pred.inputs
        predictions = []
        references = []
        metric = load_metric("squad")
        for i, (inp, pred_start, pred_end,label_start, label_end) in enumerate(zip(inputs, logits[0], logits[1], labels[0], labels[1])):
            pred_text = tokenize_func.postprocess_qa_predictions_for_train(inp, pred_start, pred_end)
            label_text = "".join(tokenize_func.tokenizer.convert_ids_to_tokens(inp[label_start:label_end+1]))
            predictions.append({"id": i, "prediction_text": pred_text})
            references.append({"id": i, "answers": label_text})
        #res = metric.compute(predictions=predictions, references=references)
        f1 = []
        for pred, gold in zip(predictions, references):
            a_gold, pred_toks = gold["answers"], pred["prediction_text"]
            f1.append(compute_f1(a_gold, pred_toks))
        return {
        #'exactmatch': res["exact_match"] / 100, # precisionはfloat
        'f-1': sum(f1)/ len(f1)
        }
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    #global tokenizer_func
    #tokenizer_func = tokenize_func
    num=0
    test_accuracy = []
    test_f1 = []
    cv = []
    kf = KFold(n_splits=5, shuffle=True, random_state=training_args.seed)
    num=0
    for train_idx, valid_idx in kf.split(train_and_valid_ds):
        num+=1
        model = AutoModelForQuestionAnswering.from_pretrained(args.model_path)
        validation_features = test_ds.map(tokenize_func.prepare_validation_features,batched=True ,remove_columns=test_ds.column_names)
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
        validation_features.set_format(type=validation_features.format["type"], columns=list(validation_features.features.keys()))
        #validation_features.to_json(file_path+'/pred_outputs/validation_features' + str(num) +'.json')
        raw_predictions = trainer.predict(validation_features)
        #validation_features.set_format(type=validation_features.format["type"], columns=list(validation_features.features.keys()))
        final_predictions = tokenize_func.postprocess_qa_predictions(test_ds, validation_features, raw_predictions.predictions)
        print(final_predictions)
        #pickle.dump(final_predictions, open(file_path+'/pred_outputs/final_predictions' + str(num) +'.pkl', "wb"))
        #pickle.dump(raw_predictions, open(file_path+'/pred_outputs/raw_predictions' + str(num) +'.pkl', "wb"))
        metric = load_metric("squad")
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in test_ds]
        #print(references)
        res = metric.compute(predictions=formatted_predictions, references=references)
        test_accuracy.append([str(num)+'th', res["exact_match"] / 100])
        f1 = []
        for pred, gold in zip(formatted_predictions, references):
            a_gold, pred_toks = gold["answers"]["text"][0], pred["prediction_text"]
            f1.append(compute_f1(a_gold, pred_toks))
        test_f1.append([str(num)+'th', sum(f1)/ len(f1)])
        cv.append(str(num)+'th')
    columns_ex = ['CV', 'EXM']
    columns_f = ['CV', 'f-1']
    test_accuracy_np = np.array([x[1] for x in test_accuracy])
    print("----------------")
    cv.append('final result')
    print(cv)
    test_accuracy.append(['final result', str(sum(test_accuracy_np)/len(test_accuracy_np))+'±'+str(np.std(test_accuracy_np))])
    test_f1_np = np.array([x[1] for x in test_f1])
    test_f1.append(['final result', str(sum(test_f1_np)/len(test_f1_np))+'±'+str(np.std(test_f1_np))])
    print(cv)
    print(test_accuracy)
    print(test_f1)
    #os.makedirs(file_path+'/pred_result', exist_ok=True)
    df = pd.DataFrame(data=test_accuracy, columns=columns_ex).set_index('CV')
    #df.to_csv(file_path+'/pred_result/evaluation_ex.csv')
    display(df)
    df = pd.DataFrame(data=test_f1, columns=columns_f).set_index('CV')
    #df.to_csv(file_path+'/pred_result/evaluation_f.csv')
    display(df)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return text.rstrip("。")

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        # do nothing
        return text

        # exclude = set(string.punctuation)
        # return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(a_gold, a_pred):
    # character-base
    gold_toks = list(normalize_answer(a_gold))
    pred_toks = list(normalize_answer(a_pred))
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

