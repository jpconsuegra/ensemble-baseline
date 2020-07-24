import re
from collections import OrderedDict
from pathlib import Path

import streamlit as st
from streamlit.ScriptRunner import StopException

from autobrat.classifier import Model
from scripts.score import compute_metrics, subtaskA, subtaskB
from scripts.utils import (
    ENTITIES,
    RELATIONS,
    Collection,
    CollectionV1Handler,
    CollectionV2Handler,
    Keyphrase,
    Relation,
    Sentence,
)

c = Collection()

if st.sidebar.checkbox("Original Data", value=False):
    c = CollectionV1Handler.load(c, Path("data/training/input_training.txt"))

if st.sidebar.checkbox("Ensemble Data", value=False):
    old_size = len(c)
    c = CollectionV2Handler.load(c, Path("data/training/ensemble.txt"))
    ensemble_size = len(c) - old_size
    top_agreement = st.sidebar.number_input(
        "Number of sentences (Ensemble)", 0, ensemble_size, ensemble_size
    )
    c.sentences = c.sentences[: old_size + top_agreement]

if st.sidebar.checkbox("Talp Data", value=False):
    old_size = len(c)
    c = CollectionV2Handler.load(c, Path("data/training/talp.txt"))
    talp_size = len(c) - old_size
    top_agreement = st.sidebar.number_input(
        "Number of sentences (TALP)", 0, talp_size, talp_size
    )
    c.sentences = c.sentences[: old_size + top_agreement]

st.write("Number of sentences", len(c))


"## Counts"

latex = ""

total = 0
for label in ENTITIES:
    n = sum(len(s.keyphrases) for s in c.filter_keyphrase([label]).sentences)
    assert n == sum(
        len([k for k in s.keyphrases if k.label == label]) for s in c.sentences
    )
    st.write(label, n)
    latex += f"\\texttt{{ {label}}}		& ${n}$ \\\\" + "\n"
    total += n

st.write("Total:", sum(len([k for k in s.keyphrases]) for s in c.sentences))

total = 0
for label in RELATIONS:
    n = sum(len(s.relations) for s in c.filter_relation([label]).sentences)
    assert n == sum(
        len([k for k in s.relations if k.label == label]) for s in c.sentences
    )
    st.write(label, n)
    latex += f"\\texttt{{ {label}}}		& ${n}$ \\\\" + "\n"
    total += n
st.write("Total:", sum(len([k for k in s.relations]) for s in c.sentences))

"## Latex"
st.text(latex)


class Baseline:
    def __init__(self):
        self.model = None

    def train(self, collection: Collection):

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

    def run(self, collection, taskA, taskB):
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


n_scenario = st.sidebar.number_input("Scenario", 1, 3, 2)
skipA = n_scenario == 3
skipB = n_scenario == 2

new_model = st.sidebar.checkbox("New model", value=False)
if new_model:
    only_for_task_a = st.sidebar.checkbox("Only for Task A", value=False)

class BaselineHandler:
    def __init__(self, baseline_A, baseline_B):
        self.baseline_A = baseline_A
        self.baseline_B = baseline_B

    def __call__(self, collection, taskA, taskB):
        if taskA:
            collection = self._do_task_A(collection)
        if taskB:
            collection = self._do_task_B(collection)
        return collection

    def _do_task_A(self, collection):
        if isinstance(self.baseline_A, Baseline):
            return self.baseline_A.run(collection, True, False)
        elif isinstance(self.baseline_A, Model):
            return self.baseline_A.predict_entities(collection.sentences)
        else:
            raise TypeError()

    def _do_task_B(self, collection):
        if isinstance(self.baseline_B, Baseline):
            return self.baseline_B.run(collection, False, True)
        elif isinstance(self.baseline_B, Model):
            return self.baseline_B.predict_relations(collection)
        else:
            raise TypeError()

if not st.sidebar.button('Do training!'):
    raise StopException()

"## Training baseline"

if not new_model or only_for_task_a:

    baseline = Baseline()
    baseline.train(c)

    keyphrases, relations = baseline.model

    "## Keyphrases"

    for item in list(keyphrases.items())[:5]:
        st.write(item)

    "## Relations"

    for item in list(relations.items())[:5]:
        st.write(item)

    baseline_A = None if new_model else baseline
    baseline_B = baseline

if new_model:

    baseline = Model(c)
    baseline.train_similarity()
    if not skipA:
        baseline.train_entities()
    if not skipB:
        baseline.train_relations()

    baseline_A = baseline
    baseline_B = baseline_B if only_for_task_a else baseline

baseline = BaselineHandler(baseline_A, baseline_B)

"## Testing"

task_name = ["main", "taskA", "taskB"][n_scenario - 1]
scenario = f"scenario{n_scenario}-{task_name}"

st.write("Scenario", scenario)

test = CollectionV1Handler.load(
    Collection(), Path(f"data/testing/{scenario}/input_scenario{n_scenario}.txt")
)
test.sentences = [s for s in test.sentences if s.annotated]

st.write("Sentences", len(test))


def eval(gold, submit, skipA, skipB):
    data = OrderedDict()

    dataA = subtaskA(gold, submit, False)
    data.update(dataA)

    if not skipB:
        dataB = subtaskB(gold, submit, dataA, False)
        data.update(dataB)

    metrics = compute_metrics(data, skipA, skipB)
    return data, metrics


empty = test.clone()
for s in empty.sentences:
    if not skipA:
        s.keyphrases = []
    if not skipB:
        s.relations = []

# if not new_model:
#     submit = baseline.run(empty, not skipA, not skipB)
# else:
#     submit = empty
#     if not skipA:
#         submit = baseline.predict_entities(submit.sentences)
#     if not skipB:
#         submit = baseline.predict_relations(submit)

submit = baseline(empty, not skipA, not skipB)

data, metrics = eval(test, submit, skipA, skipB)

st.write("$F_1$", round(metrics["f1"], 3))
st.write({k: len(v) for k, v in data.items()})
st.write(metrics)
