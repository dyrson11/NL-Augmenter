# Synonym Substitution
This perturbation adds diversity to all types of text sources (sentence, paragraph, etc.) by analyzing the training sets and making substitutions by non-frequent words.

Author name: Gerson Vizcarra (gersonw.vizcarra@gmail.com)

## What type of a transformation is this?
This transformation could augment the semantic representation of the sentence as well as test model robustness by substituting words with their non-frequent synonyms.


## What tasks does it intend to benefit?
This perturbation would benefit all tasks on text classification and generation.

###Benchmark results:

Text classification:

- We run sentiment analysis on 1000 examples of the IMDB dataset using RoBERTa (textattack/roberta-base-imdb). The original accuracy is 95.0 and the perturbed accuracy is 87.0.

- We run sentiment analysis on the 20% of the validation set (174 examples) of the SST-2 dataset using RoBERTa (textattack/roberta-base-SST-2). The original accuracy is 94.0 and the perturbed accuracy is 83.0.

- We run Natural Language Inference on 1000 examples of the validation_matched set (174 examples) of the Multi-NLI dataset using RoBERTa (roberta-large-mnli). The original accuracy is 91.0 and the perturbed accuracy is 84.0.

- We run paraphrase identification on 1000 examples of the Quora-Question-Pairs dataset using BERT (textattack/bert-base-uncased-QQP). The original accuracy is 92.0 and the perturbed accuracy is 82.0.

## Related Work
The tokenization and POS tagging were done using Spacy

The synonyms are based on WordNet via NLTK

```bibtex
@book{miller1998wordnet,
  title={WordNet: An electronic lexical database},
  author={Miller, George A},
  year={1998},
  publisher={MIT press}
}
@inproceedings{bird2006nltk,
  title={NLTK: the natural language toolkit},
  author={Bird, Steven},
  booktitle={Proceedings of the COLING/ACL 2006 Interactive Presentation Sessions},
  pages={69--72},
  year={2006}
}
```


## What are the limitations of this transformation?
The synonyms space is dependent on WordNet and could be limited. The current implementation could generate misspelled sentences.
