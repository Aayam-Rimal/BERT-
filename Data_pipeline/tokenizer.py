import os

os.environ["RAYON_NUM_THREADS"] = (
    "2"  
)

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.trainers import WordPieceTrainer

wiki_stream = load_dataset(
    "wikimedia/wikipedia", "20231101.en", split="train", streaming=True
)



tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.normalizer = BertNormalizer(lowercase=True)
tokenizer.pre_tokenizer = BertPreTokenizer()


trainer= WordPieceTrainer(vocab_size=32000,min_frequency=5,special_tokens=["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"],limit_alphabet=1000)

tokenizer.train(
    files=["wiki_500k.txt"],
    trainer=trainer
)

tokenizer.save("bert_tokenizer.json")