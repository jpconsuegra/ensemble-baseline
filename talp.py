from pathlib import Path
from scripts.utils import Collection, CollectionV1Handler, CollectionV2Handler

talp = CollectionV1Handler.load(
    Collection(), Path("data/training/talp-576640/scenario1-main/input_scenario1.txt")
)
print(f"Talp: {len(talp)}")

ensemble = CollectionV2Handler.load(Collection(), Path("data/training/ensemble.txt"))
print(f"Ensemble: {len(ensemble)}")

sentences = set([s.text for s in ensemble.sentences])

selection = Collection([s for s in talp.sentences if s.text in sentences])
print(f"Selection: {len(selection)}")

output = Path("data/training/talp.txt")
output.parent.mkdir(exist_ok=True)

CollectionV2Handler.dump(selection, output, skip_empty_sentences=False)