from tokenizers import Tokenizer
from bert.MLM import MLM
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

tokenizer_file = BASE_DIR / "bert_tokenizer.json"

class Dataset:

    def __init__(self,data,tokenizer_file):

        self.data= data
        self.tokenizer= Tokenizer.from_file(str(tokenizer_file))

        with open(tokenizer_file,"r") as f:
           tokenizer_config= json.load(f)

        self.vocab= tokenizer_config["model"]["vocab"]
        self.vocab_size= len(self.vocab)

        self.cls_id= self.tokenizer.token_to_id("[CLS]")
        self.sep_id= self.tokenizer.token_to_id("[SEP]")
        self.mask_id= self.tokenizer.token_to_id("[MASK]")

        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

        self.special_tokens_id = [
            self.tokenizer.token_to_id(token)
            for token in special_tokens
        ]



    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        src= self.data[index]        

        src_id= self.tokenizer.encode(src).ids

        input_ids= [self.cls_id] + src_id + [self.sep_id]

        Mask_ids,label_ids= MLM(input_ids,self.mask_id,special_tokens_id=self.special_tokens_id,vocab=self.vocab_size)

        return Mask_ids,label_ids


if __name__=="__main__":
    data= ["hello he is a man","i can not recall every moment"]

    dataset= Dataset(data,tokenizer_file)

    print(dataset[0])