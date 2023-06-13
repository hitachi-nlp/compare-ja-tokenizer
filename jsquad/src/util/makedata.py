import traceback
from datasets import load_dataset, Dataset, DatasetDict
from IPython.core.display import display
from tqdm.auto import tqdm
import collections
import torch
import numpy as np

# preprocessing
def make_data(tokenizer_func, args):
    dataset = load_dataset('json', data_files={
            'train': args.train_data_path,
            'valid': args.val_data_path,
        })
    dataset = re_makedataset(dataset)
    display(dataset)
    tokenized_datasets = DatasetDict()
    tokenized_datasets["train"] = dataset["train"].map(tokenizer_func.prepare_train_features, remove_columns=dataset["train"].column_names, batched=True)
    tokenized_datasets["valid"] = dataset["valid"]
    return tokenized_datasets['train'].shuffle(seed=42), tokenized_datasets['valid'].shuffle(seed=42)

def re_makedataset(dataset):
    dic = {}
    for spec in dataset:
        contexts = []
        questions = []
        answers = []
        ids = []
        titles = []
        for data in dataset[spec]["data"][0]:
            title = data["title"]
            for data2 in data["paragraphs"]:
                context = data2["context"]
                for data3 in data2["qas"]:
                    contexts.append(context)
                    questions.append(data3["question"])
                    answers.append({"text":[data3["answers"][0]["text"]], 'answer_start':[data3["answers"][0]["answer_start"]]})
                    ids.append(data3["id"])
                    titles.append(title)
        dic[spec] = {"context":contexts, "question":questions, "answers":answers, "id":ids, "title":titles}
        
    dataset_train = Dataset.from_dict(dic["train"])
    dataset_valid = Dataset.from_dict(dic["valid"])
    return DatasetDict({"train": dataset_train, "valid":dataset_valid})



# tokenize class
class TokenizerFuncClass():
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def prepare_train_features(self, examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace

        # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples["context"],
            examples["question"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        sample_list = list(range(1000))
        import collections
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples["offset_mapping"]

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        for i, offsets in enumerate(offset_mapping):
            ct = 0
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 0:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 0:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                    #tokenized_examples["decoded_ans"].append("")
                    #tokenized_examples["true_ans"].append("")
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        #print(offsets[token_end_index][1])
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
                    #text = self.tokenizer.decode(input_ids[token_start_index - 1:token_end_index + 2])
                    #tokenized_examples["decoded_ans"].append(text)
                    #tokenized_examples["true_ans"].append(answers["text"][0])
        return tokenized_examples

    def prepare_validation_features(self, examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples["context"],
            examples["question"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # We keep the example_id that gave us this feature and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 0
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    def postprocess_qa_predictions(self, examples, features, raw_predictions, n_best_size = 20, max_answer_length = 30):
        all_start_logits, all_end_logits = raw_predictions
        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        # The dictionaries we have to fill.
        predictions = collections.OrderedDict()

        # Logging.
        print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

        # Let's loop over all the examples!
        for example_index, example in enumerate(tqdm(examples)):
            # Those are the indices of the features associated to the current example.
            feature_indices = features_per_example[example_index]
            min_null_score = None # Only used if squad_v2 is True.
            valid_answers = []
            
            context = example["context"]
            # Looping through all the features associated to the current example.
            for feature_index in feature_indices:
                # We grab the predictions of the model for this feature.
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]
                # This is what will allow us to map some the positions in our logits to span of texts in the original
                # context.
                offset_mapping = features[feature_index]["offset_mapping"]

                # Update minimum null prediction.
                cls_index = features[feature_index]["input_ids"].index(self.tokenizer.cls_token_id)
                feature_null_score = start_logits[cls_index] + end_logits[cls_index]
                if min_null_score is None or min_null_score < feature_null_score:
                    min_null_score = feature_null_score

                # Go through all possibilities for the `n_best_size` greater start and end logits.
                start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
                end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                        # to part of the input_ids that are not in the context.
                        if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                        ):
                            continue
                        # Don't consider answers with a length that is either < 0 or > max_answer_length.
                        if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                            continue

                        start_char = offset_mapping[start_index][0]
                        end_char = offset_mapping[end_index][1]
                        valid_answers.append(
                            {
                                "score": start_logits[start_index] + end_logits[end_index],
                                "text": context[start_char: end_char]
                            }
                        #print()
                        )
            
            if len(valid_answers) > 0:
                best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
            else:
                # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
                # failure.
                best_answer = {"text": "", "score": 0.0}
            
            # Let's pick our final answer: the best one or the null answer (only for squad_v2)
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            predictions[example["id"]] = answer
        return predictions

    def postprocess_qa_predictions_for_train(self, inp, start_logits, end_logits, n_best_size = 20, max_answer_length = 30):
        # The dictionaries we have to fill.
        # Logging.
        #print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")
        # Those are the indices of the features associated to the current example.
        min_null_score = None # Only used if squad_v2 is True.
        valid_answers = []

        # This is what will allow us to map some the positions in our logits to span of texts in the original
        # context.

        # Update minimum null prediction.
        cls_index = list(inp).index(self.tokenizer.cls_token_id)
        feature_null_score = start_logits[cls_index] + end_logits[cls_index]
        if min_null_score is None or min_null_score < feature_null_score:
            min_null_score = feature_null_score
        original_offset_mapping = self.get_original_offset_mapping(inp)
        context_range = self.get_context(inp)
        # Go through all possibilities for the `n_best_size` greater start and end logits.
        start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
        end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
        for start_index in start_indexes:
            for end_index in end_indexes:
                # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                # to part of the input_ids that are not in the context.
                if (
                    start_index >= context_range
                    or end_index >= context_range
                    or original_offset_mapping[start_index] is None
                    or original_offset_mapping[end_index] is None
                ):
                    continue
                # Don't consider answers with a length that is either < 0 or > max_answer_length.
                if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                    continue
                start_char = original_offset_mapping[start_index][0]
                end_char = original_offset_mapping[end_index][1]
                valid_answers.append(
                    {
                        "score": start_logits[start_index] + end_logits[end_index],
                        "text": "".join(self.tokenizer.convert_ids_to_tokens(inp)).replace("##", "")[start_char: end_char]
                    }
                )
        
            if len(valid_answers) > 0:
                best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
            else:
                # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
                # failure.
                best_answer = {"text": "", "score": 0.0}
        
        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
        predictions = answer

        return predictions

    def get_context(self, inp):
        return [i for i, x in enumerate(list(inp)) if x == self.tokenizer.sep_token_id][1]

    def get_original_offset_mapping(self, inp):
        original_offset_mapping = []
        start = 0
        for i in range(len(inp)):
            self.tokenizer.convert_ids_to_tokens(int(inp[i]))
            token_string =  self.tokenizer.convert_ids_to_tokens(int(inp[i]))
            token_length = len(token_string.replace("##", ""))
            original_offset_mapping.append([start, start + token_length])
            start += token_length
        return original_offset_mapping