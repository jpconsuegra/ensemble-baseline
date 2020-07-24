# coding: utf8

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import List

from scripts.submit import Algorithm, Run, handle_args
from scripts.utils import (
    Collection,
    Keyphrase,
    Relation,
    Sentence,
    CollectionV1Handler,
    CollectionV2Handler,
)


class Baseline(Algorithm):
    def __init__(self):
        self.model = None

    def train(self, finput: Path):

        collection = Collection()
        collection = CollectionV1Handler.load(collection, finput / "input_training.txt")
        # collection = CollectionV2Handler.load(collection, finput / 'ensemble.txt')

        self.model = keyphrases, relations = {}, {}

        for sentence in collection.sentences:
            for keyphrase in sentence.keyphrases:
                text = keyphrase.text.lower()
                keyphrases[text] = keyphrase.label

        for sentence in collection.sentences:
            for relation in sentence.relations:
                origin = relation.from_phrase
                origin_text = origin.text.lower()
                destination = relation.to_phrase
                destination_text = destination.text.lower()

                relations[
                    origin_text, origin.label, destination_text, destination.label
                ] = relation.label

    def run(self, collection, *args, taskA, taskB, **kargs):
        gold_keyphrases, gold_relations = self.model

        if taskA:
            next_id = 0
            for gold_keyphrase, label in gold_keyphrases.items():
                for sentence in collection.sentences:
                    text = sentence.text.lower()
                    pattern = r"\b" + gold_keyphrase + r"\b"
                    for match in re.finditer(pattern, text):
                        keyphrase = Keyphrase(sentence, label, next_id, [match.span()])
                        keyphrase.split()
                        next_id += 1

                        sentence.keyphrases.append(keyphrase)

        if taskB:
            for sentence in collection.sentences:
                for origin in sentence.keyphrases:
                    origin_text = origin.text.lower()
                    for destination in sentence.keyphrases:
                        destination_text = destination.text.lower()
                        try:
                            label = gold_relations[
                                origin_text,
                                origin.label,
                                destination_text,
                                destination.label,
                            ]
                        except KeyError:
                            continue
                        relation = Relation(sentence, origin.id, destination.id, label)
                        sentence.relations.append(relation)

                sentence.remove_dup_relations()

        return collection


def main():
    baseline = Baseline()
    baseline.train(Path("data/training/"))

    collection = CollectionV1Handler.load_dir(
        Collection(), Path("data/testing/scenario2-taskA/")
    )
    output = baseline.run(collection, taskA=True, taskB=False)
    CollectionV1Handler.dump(output, Path("pepe/input_scenario2.txt"), False)


if __name__ == "__main__":
    main()
