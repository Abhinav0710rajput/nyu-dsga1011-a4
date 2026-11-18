import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

from preprocess import preprocess_nl, preprocess_sql

PAD_IDX = 0
MAX_LEN = 512  # longest SQL target is 511 tokens incl. </s> (see Q4)

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        # Decoder beginning-of-sentence token: <extra_id_0>
        self.bos_id = self.tokenizer.convert_tokens_to_ids('<extra_id_0>')
        self.encoder_ids, self.decoder_ids = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        # Pre-processing (task prefix, whitespace, '<' placeholder) lives in preprocess.py (see Q4)
        nl_lines = load_lines(os.path.join(data_folder, f'{split}.nl'))
        nl_lines = [preprocess_nl(line) for line in nl_lines]
        encoder_ids = [tokenizer(line, truncation=True, max_length=MAX_LEN)['input_ids'] for line in nl_lines]

        # The test set has no SQL targets
        if split == 'test':
            return encoder_ids, None

        sql_lines = load_lines(os.path.join(data_folder, f'{split}.sql'))
        sql_lines = [preprocess_sql(line) for line in sql_lines]
        # Tokenizer appends </s> (EOS) to every target
        decoder_ids = [tokenizer(line, truncation=True, max_length=MAX_LEN)['input_ids'] for line in sql_lines]
        return encoder_ids, decoder_ids

    def __len__(self):
        return len(self.encoder_ids)

    def __getitem__(self, idx):
        encoder_ids = torch.tensor(self.encoder_ids[idx], dtype=torch.long)
        if self.decoder_ids is None:
            return encoder_ids, self.bos_id
        # Teacher forcing: decoder input is <BOS> + target[:-1], decoder target is the full target (ends in </s>)
        target = torch.tensor(self.decoder_ids[idx], dtype=torch.long)
        decoder_input = torch.cat([torch.tensor([self.bos_id], dtype=torch.long), target[:-1]])
        return encoder_ids, decoder_input, target, self.bos_id

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    decoder_inputs = pad_sequence([item[1] for item in batch], batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence([item[2] for item in batch], batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.tensor([[item[3]] for item in batch], dtype=torch.long)  # B x 1
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.tensor([[item[1]] for item in batch], dtype=torch.long)  # B x 1
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x