import re
import nltk
import spacy
from nltk.corpus import wordnet as wn
import numpy as np

from interfaces.SentenceOperation import SentenceOperation
from tasks.TaskTypes import TaskType


"""
Base Class for implementing the different input transformations a generation should be robust against.
"""


# TODO: evaluate whether it is necessary to separate the wordnet vocabulary
def get_low_frequency_words(data, spacy_pipeline, min_appearances):
    print("\t Finding the low frequency words")
    print("\t\t Tokenizing the training dataset")
    print("\t\t Training data has {} examples".format(len(data)))
    docs = spacy_pipeline.pipe(data, batch_size=200)
    vocabulary = {k: 0 for k in wn.all_lemma_names()}
    print("\t\t Counting words")
    vocabulary_list = set(vocabulary.keys())
    for doc in docs:
        for token in doc:
            word = token.text
            if word in vocabulary_list:
                vocabulary[word] += 1
    print("\t\t Evaluating low appearances")
    low_frequency_words = set()
    for word in vocabulary.keys():
        if vocabulary[word] <= min_appearances:
            low_frequency_words.add(word)

    print(
        "\t Found {}/{} low frequency words".format(
            len(low_frequency_words), len(list(vocabulary.keys()))
        )
    )
    return low_frequency_words


def untokenize(words):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    ref: https://github.com/commonsense/metanl/blob/master/metanl/token_utils.py#L28
    """
    text = " ".join(words)
    step1 = (
        text.replace("`` ", '"').replace(" ''", '"').replace(". . .", "...")
    )
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r" ([.,:;?!%]+)$", r"\1", step3)
    step5 = (
        step4.replace(" '", "'")
        .replace(" n't", "n't")
        .replace("can not", "cannot")
    )
    step6 = step5.replace(" ` ", " '")
    return step6.strip()


def pos_to_wordnet_pos(penntag, returnNone=False):
    """Mapping from POS tag word wordnet pos tag"""
    morphy_tag = {"NN": wn.NOUN, "JJ": wn.ADJ, "VB": wn.VERB, "RB": wn.ADV}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return None if returnNone else ""


def get_synonyms(word, pos):
    "Gets word synonyms for part of speech"
    for synset in wn.synsets(word, pos=pos_to_wordnet_pos(pos)):
        for lemma in synset.lemmas():
            yield lemma.name()


def synonym_substitution(
    text, spacy_pipeline, low_frequency_words, seed=42, prob=0.5
):
    np.random.seed(seed)
    results = []
    doc = spacy_pipeline(text)
    result = []
    for token in doc:
        word = token.text
        tag_type = token.tag_
        synonyms = sorted(
            set(
                synonym
                for synonym in get_synonyms(word, tag_type)
                if synonym.lower() != word.lower()
            )
        )
        synonyms = list(low_frequency_words.intersection(synonyms))
        if len(synonyms) > 0 and np.random.random() < prob:
            result.append(np.random.choice(synonyms).replace("_", " "))
        else:
            result.append(word)

    result = untokenize(result)
    if result not in results:
        # make sure there is no dup in results
        results.append(result)
    return results


"""
Substitute words with synonyms using stanza (for POS) and wordnet via nltk (for synonyms)
"""


class OutOfVocabularySubstitutions(SentenceOperation):
    tasks = [
        TaskType.TEXT_CLASSIFICATION,
        TaskType.TEXT_TO_TEXT_GENERATION,
    ]
    languages = ["en"]

    def __init__(self, seed=42, prob=0.5, min_appearances=0):
        self.nlp = spacy.load(
            "en_core_web_sm",
            disable=["parser", "ner", "lemmatizer", "textcat"],
        )
        self.prob = prob
        nltk.download("wordnet")
        self.min_appearances = min_appearances

    def generate(self, text):
        perturbed = synonym_substitution(
            text=text,
            spacy_pipeline=self.nlp,
            low_frequency_words=self.low_frequency_words,
        )
        return perturbed

    def analyze(self, training_data, training_labels=None):
        self.low_frequency_words = get_low_frequency_words(
            training_data, self.nlp, self.min_appearances
        )
