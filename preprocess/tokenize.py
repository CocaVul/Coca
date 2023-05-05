from argparse import ArgumentParser
from os import listdir, remove
from os.path import splitext, join, isdir, exists

import youtokentome as yttm
from omegaconf import DictConfig, OmegaConf


def tokenize(config: DictConfig):
    data_path = join(config.data_folder, config.dataset.name, config.dataset.dir)
    model_path = join(data_path, config.dataset.tokenizer_name)
    buffer_path = "text.yttm"
    if exists(buffer_path):
        remove(buffer_path)

    for file in listdir(join(data_path, config.train_holdout)):
        transformed_files_path = join(data_path, config.train_holdout, file)
        if isdir(transformed_files_path):
            for transformed_file in listdir(transformed_files_path):
                file_path = join(transformed_files_path, transformed_file)
                _, ext = splitext(file_path)
                if ext in [".cpp", ".c"]:
                    with open(file_path, "r", encoding="utf8", errors='ignore') as file_:
                        text = file_.read() + "\n"
                        with open(buffer_path, "a") as buffer_:
                            buffer_.write(text)

    _ = yttm.BPE.train(
        data="text.yttm",
        model=model_path,
        pad_id=config.dataset.pad_id,
        unk_id=config.dataset.unk_id,
        bos_id=config.dataset.bos_id,
        eos_id=config.dataset.eos_id,
        vocab_size=config.dataset.vocab_size,
        n_threads=config.num_workers
    )

    remove("text.yttm")


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config_path", type=str)
    args = arg_parser.parse_args()
    config_ = OmegaConf.load(args.config_path)
    tokenize(config=config_)
