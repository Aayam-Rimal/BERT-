from .dataset import Dataset
import torch
from torch.utils.data import DataLoader
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

tokenizer_file = BASE_DIR / "bert_tokenizer.json"


def pad_sequence(seq,max_length=512,pad_id=-100):

    return seq + [pad_id] * (max_length-len(seq))

def collate(batch,max_length=512):

    src,label= zip(*batch)

    src= [x[:max_length] for x in src]
    label= [x[:max_length] for x in label]

    src= [pad_sequence(x,max_length) for x in src]
    label= [pad_sequence(x,max_length) for x in label]

    return {
        "src": torch.tensor(src, dtype=torch.long),
        "label": torch.tensor(label, dtype=torch.long)
    }



if __name__=="__main__":
    data = [
        "my name is mouse and my house is in america",
        "he is a hero who was once a zero",
        "can we all not do that",
        "the fox jumped over the dog"

    ]
    dataset= Dataset(data,tokenizer_file)

    train_loader= DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate
    )

    print(next(iter(train_loader)))