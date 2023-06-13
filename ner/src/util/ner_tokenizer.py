import itertools
import numpy as np
import torch
from transformers import PreTrainedTokenizerFast
from .mecab_pretokenizer import MecabPreTokenizer

mecab_pretokenizer = MecabPreTokenizer()

class NER_tokenizer_BIO(PreTrainedTokenizerFast):

    def encode_plus_tagged(self, text, entities, max_length):
        self.num_entity_type = 8
        splitted = []
        position = 0
        for entity in entities:
            start = entity['span'][0]
            end = entity['span'][1]
            label = entity['type_id']
            splitted.append({'text':text[position:start], 'label':0})
            splitted.append({'text':text[start:end], 'label':label})
            position = end
        splitted.append({'text': text[position:], 'label':0})
        splitted = [ s for s in splitted if s['text'] ]

        tokens = []
        labels = []
        for s in splitted:
            tokens_splitted = self.tokenize(s['text'])
            label = s['label']
            if label > 0 and len(tokens_splitted) > 0:
                labels_splitted =  \
                    [ label + self.num_entity_type ] * len(tokens_splitted)
                labels_splitted[0] = label
            else:
                labels_splitted =  [0] * len(tokens_splitted)
            
            tokens.extend(tokens_splitted)
            labels.extend(labels_splitted)

        input_ids = self.convert_tokens_to_ids(tokens)
        encoding = self.prepare_for_model(
            input_ids, 
            max_length=max_length, 
            padding='max_length',
            truncation=True
        ) 

        labels = [0] + labels[:max_length-2] + [0]
        labels = labels + [0]*( max_length - len(labels) )
        encoding['labels'] = labels

        return encoding

    def encode_plus_untagged(
        self, text, max_length=None, return_tensors=None
    ):
        tokens = []
        tokens_original = []
        words = self._tokenizer.pre_tokenizer.pre_tokenize_str(text)
        words = [word[0] for word in words]
        for word in words:
            tokens_word = self.tokenize(word) 
            tokens.extend(tokens_word)
            if len(tokens_word) == 0:
                pass
            else:
                if tokens_word[0] == '[UNK]':
                    tokens_original.append(word)
                else:
                    tokens_original.extend([
                        token.replace('##','') for token in tokens_word
                    ])

        position = 0
        spans = []
        for token in tokens_original:
            l = len(token)
            while 1:
                if token != text[position:position+l]:
                    position += 1
                else:
                    spans.append([position, position+l])
                    position += l
                    break

        input_ids = self.convert_tokens_to_ids(tokens) 
        encoding = self.prepare_for_model(
            input_ids, 
            max_length=max_length, 
            padding='max_length' if max_length else False, 
            truncation=True if max_length else False
        )
        sequence_length = len(encoding['input_ids'])
        spans = [[-1, -1]] + spans[:sequence_length-2] 
        spans = spans + [[-1, -1]] * ( sequence_length - len(spans) ) 

        if return_tensors == 'pt':
            encoding = { k: torch.tensor([v]) for k, v in encoding.items() }

        return encoding, spans

    @staticmethod
    def Viterbi(scores_bert, num_entity_type, penalty=10000):
        m = 2*num_entity_type + 1
        penalty_matrix = np.zeros([m, m])
        for i in range(m):
            for j in range(1+num_entity_type, m):
                if not ( (i == j) or (i+num_entity_type == j) ): 
                    penalty_matrix[i,j] = penalty
        
        path = [ [i] for i in range(m) ]
        if len(scores_bert) == 0:
            labels_optimal = []
            pass
        else:
            scores_path = scores_bert[0] - penalty_matrix[0,:]
            scores_bert = scores_bert[1:]

            for scores in scores_bert:
                assert len(scores) == 2*num_entity_type + 1
                score_matrix = np.array(scores_path).reshape(-1,1) \
                    + np.array(scores).reshape(1,-1) \
                    - penalty_matrix
                scores_path = score_matrix.max(axis=0)
                argmax = score_matrix.argmax(axis=0)
                path_new = []
                for i, idx in enumerate(argmax):
                    path_new.append( path[idx] + [i] )
                path = path_new

            labels_optimal = path[np.argmax(scores_path)]
        return labels_optimal

    def convert_bert_output_to_entities(self, text, scores, spans):
        assert len(spans) == len(scores)
        num_entity_type = self.num_entity_type
        
        scores = [score for score, span in zip(scores, spans) if span[0]!=-1]
        spans = [span for span in spans if span[0]!=-1]

        labels = self.Viterbi(scores, num_entity_type)

        entities = []
        for label, group \
            in itertools.groupby(enumerate(labels), key=lambda x: x[1]):
            
            group = list(group)
            start = spans[group[0][0]][0]
            end = spans[group[-1][0]][1]

            if label != 0:
                if 1 <= label <= num_entity_type:
                    entity = {
                        "name": text[start:end],
                        "span": [start, end],
                        "type_id": label
                    }
                    entities.append(entity)
                else:
                    entity['span'][1] = end 
                    entity['name'] = text[entity['span'][0]:entity['span'][1]]
                
        return entities


from transformers import BertJapaneseTokenizer

class NER_tokenizer_BIO_tohoku(BertJapaneseTokenizer):
    def __init__(self, *args, **kwargs):
        self.num_entity_type = kwargs.pop('num_entity_type')
        super().__init__(*args, **kwargs)

    def encode_plus_tagged(self, text, entities, max_length):
        splitted = []
        position = 0
        for entity in entities:
            start = entity['span'][0]
            end = entity['span'][1]
            label = entity['type_id']
            splitted.append({'text':text[position:start], 'label':0})
            splitted.append({'text':text[start:end], 'label':label})
            position = end
        splitted.append({'text': text[position:], 'label':0})
        splitted = [ s for s in splitted if s['text'] ]

        tokens = []
        labels = []
        for s in splitted:
            tokens_splitted = self.tokenize(s['text'])
            label = s['label']
            if label > 0:
                labels_splitted =  \
                    [ label + self.num_entity_type ] * len(tokens_splitted)
                labels_splitted[0] = label
            else:
                labels_splitted =  [0] * len(tokens_splitted)
            
            tokens.extend(tokens_splitted)
            labels.extend(labels_splitted)

        input_ids = self.convert_tokens_to_ids(tokens)
        encoding = self.prepare_for_model(
            input_ids, 
            max_length=max_length, 
            padding='max_length',
            truncation=True
        ) 

        labels = [0] + labels[:max_length-2] + [0]
        labels = labels + [0]*( max_length - len(labels) )
        encoding['labels'] = labels

        return encoding

    def encode_plus_untagged(
        self, text, max_length=None, return_tensors=None
    ):
        tokens = []
        tokens_original = []
        words = self.word_tokenizer.tokenize(text)
        for word in words:
            tokens_word = self.subword_tokenizer.tokenize(word) 
            tokens.extend(tokens_word)
            if tokens_word[0] == '[UNK]':
                tokens_original.append(word)
            else:
                tokens_original.extend([
                    token.replace('##','') for token in tokens_word
                ])

        position = 0
        spans = []
        for token in tokens_original:
            l = len(token)
            while 1:
                if token != text[position:position+l]:
                    position += 1
                else:
                    spans.append([position, position+l])
                    position += l
                    break

        input_ids = self.convert_tokens_to_ids(tokens) 
        encoding = self.prepare_for_model(
            input_ids, 
            max_length=max_length, 
            padding='max_length' if max_length else False, 
            truncation=True if max_length else False
        )
        sequence_length = len(encoding['input_ids'])
        spans = [[-1, -1]] + spans[:sequence_length-2] 
        spans = spans + [[-1, -1]] * ( sequence_length - len(spans) ) 

        if return_tensors == 'pt':
            encoding = { k: torch.tensor([v]) for k, v in encoding.items() }

        return encoding, spans

    @staticmethod
    def Viterbi(scores_bert, num_entity_type, penalty=10000):
        m = 2*num_entity_type + 1
        penalty_matrix = np.zeros([m, m])
        for i in range(m):
            for j in range(1+num_entity_type, m):
                if not ( (i == j) or (i+num_entity_type == j) ): 
                    penalty_matrix[i,j] = penalty
        
        path = [ [i] for i in range(m) ]
        scores_path = scores_bert[0] - penalty_matrix[0,:]
        scores_bert = scores_bert[1:]

        for scores in scores_bert:
            assert len(scores) == 2*num_entity_type + 1
            score_matrix = np.array(scores_path).reshape(-1,1) \
                + np.array(scores).reshape(1,-1) \
                - penalty_matrix
            scores_path = score_matrix.max(axis=0)
            argmax = score_matrix.argmax(axis=0)
            path_new = []
            for i, idx in enumerate(argmax):
                path_new.append( path[idx] + [i] )
            path = path_new

        labels_optimal = path[np.argmax(scores_path)]
        return labels_optimal

    def convert_bert_output_to_entities(self, text, scores, spans):
        assert len(spans) == len(scores)
        num_entity_type = self.num_entity_type
        
        scores = [score for score, span in zip(scores, spans) if span[0]!=-1]
        spans = [span for span in spans if span[0]!=-1]

        labels = self.Viterbi(scores, num_entity_type)

        entities = []
        for label, group \
            in itertools.groupby(enumerate(labels), key=lambda x: x[1]):
            
            group = list(group)
            start = spans[group[0][0]][0]
            end = spans[group[-1][0]][1]

            if label != 0:
                if 1 <= label <= num_entity_type:
                    entity = {
                        "name": text[start:end],
                        "span": [start, end],
                        "type_id": label
                    }
                    entities.append(entity)
                else:
                    entity['span'][1] = end 
                    entity['name'] = text[entity['span'][0]:entity['span'][1]]
                
        return entities