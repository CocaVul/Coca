import pickle
from collections import Counter
from os import remove
from os.path import join, dirname, exists
from random import shuffle
from typing import Counter as CounterType, Type, Dict

from code2seq.data.vocabulary import Vocabulary
from commode_utils.filesystem import count_lines_in_file
from commode_utils.vocabulary import BaseVocabulary
from tqdm import tqdm


def build_code2seq_vocab(
    train_data: str,
    test_data: str,
    val_data: str,
    vocabulary_cls: Type[BaseVocabulary] = Vocabulary
):
    counters: Dict[str, CounterType[str]] = {
        key: Counter() for key in [vocabulary_cls.LABEL, vocabulary_cls.TOKEN, vocabulary_cls.NODE]
    }
    with open(train_data, "r") as f_in:
        for raw_sample in tqdm(f_in, total=count_lines_in_file(train_data)):
            vocabulary_cls.process_raw_sample(raw_sample, counters)

    for data in [test_data, val_data]:
        with open(data, "r") as f_in:
            for raw_sample in tqdm(f_in, total=count_lines_in_file(data)):
                label, *_ = raw_sample.split(" ")
                counters[vocabulary_cls.LABEL].update(label.split(vocabulary_cls._separator))

    for feature, counter in counters.items():
        print(f"Count {len(counter)} {feature}, top-5: {counter.most_common(5)}")

    dataset_dir = dirname(train_data)
    vocabulary_file = join(dataset_dir, vocabulary_cls.vocab_filename)
    with open(vocabulary_file, "wb") as f_out:
        pickle.dump(counters, f_out)


def _get_id2value_from_csv(path_: str) -> Dict[str, str]:
    with open(path_, "r") as f:
        lines = f.read().strip().split("\n")[1:]
        lines = [line.split(",", maxsplit=1) for line in lines]
        return {k: v for k, v in lines}


def process_astminer_csv(data_folder: str, dataset_name: str, holdout_name: str, is_shuffled: bool):
    """
    Preprocessing for files tokens.csv, paths.csv, node_types.csv
    """
    dataset_path = join(data_folder, dataset_name)
    id_to_token_data_path = join(dataset_path, f"tokens.{holdout_name}.csv")
    id_to_type_data_path = join(dataset_path, f"node_types.{holdout_name}.csv")
    id_to_paths_data_path = join(dataset_path, f"paths.{holdout_name}.csv")
    path_contexts_path = join(dataset_path, f"path_contexts.{holdout_name}.csv")
    output_c2s_path = join(dataset_path, f"{dataset_name}.{holdout_name}.c2s")

    id_to_paths_stored = _get_id2value_from_csv(id_to_paths_data_path)
    id_to_paths = {index: [n for n in nodes.split()] for index, nodes in id_to_paths_stored.items()}

    id_to_node_types = _get_id2value_from_csv(id_to_type_data_path)
    id_to_node_types = {index: node_type.rsplit(" ", maxsplit=1)[0] for index, node_type in
                        id_to_node_types.items()}

    id_to_tokens = _get_id2value_from_csv(id_to_token_data_path)

    if exists(output_c2s_path):
        remove(output_c2s_path)
    with open(path_contexts_path, "r") as path_contexts_file, open(output_c2s_path, "a+") as c2s_output:
        output_lines = []
        for line in tqdm(path_contexts_file, total=count_lines_in_file(path_contexts_path)):
            label, *path_contexts = line.split()
            parsed_line = [label]
            for path_context in path_contexts:
                from_token_id, path_types_id, to_token_id = path_context.split(",")
                from_token, to_token = id_to_tokens[from_token_id], id_to_tokens[to_token_id]
                if (" " in from_token) or (" " in to_token) or ():
                    continue
                nodes = [id_to_node_types[p_] for p_ in id_to_paths[path_types_id]]
                for node in nodes:
                    if " " in node:
                        break
                parsed_line.append(",".join([from_token, "|".join(nodes), to_token]))
            output_lines.append(" ".join(parsed_line + ["\n"]))
        if is_shuffled:
            shuffle(output_lines)
        c2s_output.write("".join(output_lines))
