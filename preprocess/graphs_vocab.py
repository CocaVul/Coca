import json
from os import listdir
from os.path import join, isdir, isfile

from omegaconf import DictConfig
from tqdm import tqdm


def is_json_file(path: str):
    ext = path.rsplit(".", 1)[-1]
    return isfile(path) and (ext == "json")


def build_graphs_vocab(config: DictConfig):
    graphs_storage = join(config.data_folder, config.dataset.name, config.dataset.dir)

    edges_types = set()
    vertexes_types = set()
    vertexes_names = set()

    holdout_path = join(graphs_storage, config.train_holdout)

    for class_ in tqdm(listdir(holdout_path)):
        class_path = join(holdout_path, class_)
        if isdir(class_path):
            paths = [
                join(class_path, file) for file in listdir(class_path) if is_json_file(join(class_path, file))
            ]

            for graph_path in tqdm(paths):
                with open(graph_path, "r") as f:
                    graph = json.load(f)
                e = json.loads(graph["edges"])
                v = json.loads(graph["vertexes"])
                vertexes_types.update(set(v_["label"] for v_ in v))
                vertexes_names.update(set(v_["name"] for v_ in v))
                edges_types.update(set(e_["label"] for e_ in e))

    vertexes_types.add("UNKNOWN")
    vertexes_types = sorted(list(vertexes_types))
    edges_types.add("UNKNOWN")
    edges_types = sorted(list(edges_types))
    vocab = {
        "v_type2id": {v_type: id_ for id_, v_type in enumerate(vertexes_types)},
        "v_name2id": {v_name: id_ for id_, v_name in enumerate(vertexes_names)},
        "e_type2id": {e_type: id_ for id_, e_type in enumerate(edges_types)}
    }

    with open(join(graphs_storage, config.dataset.vocab_file), "w") as vocab_f:
        json.dump(vocab, vocab_f)
