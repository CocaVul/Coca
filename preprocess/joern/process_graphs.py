import multiprocessing
import subprocess
from argparse import ArgumentParser
from os import listdir
from os import mkdir
from os.path import exists, join, isdir, isfile

from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

c_family_exts = ["c", "cpp"]


def _is_c_family_file(path: str):
    ext = path.rsplit(".", 1)[-1]
    return isfile(path) and (ext in c_family_exts)


def run_joern_parse(file: str, class_path: str, class_cpg: str):
    file_path = join(class_path, file)
    base_name = file.rsplit('.', 1)[0]
    cpg_file_path = join(class_cpg, f"{base_name}.bin")

    # joern-parse
    if not exists(cpg_file_path):
        subprocess.check_call([
            "joern-parse",
            file_path, "--out",
            cpg_file_path
        ])


def process_graphs(config: DictConfig):
    data_path = join(config.data_folder, config.dataset.name, "raw")
    graphs_path = join(config.data_folder, config.dataset.name, config.dataset.dir)
    cpg_path = join(config.data_folder, config.dataset.name, "cpg")
    num_cores = multiprocessing.cpu_count()

    if not exists(graphs_path):
        mkdir(graphs_path)

    if not exists(cpg_path):
        mkdir(cpg_path)

    for holdout in [config.train_holdout, config.val_holdout, config.test_holdout]:
        holdout_path = join(data_path, holdout)
        holdout_output = join(graphs_path, holdout)

        if not exists(holdout_output):
            mkdir(holdout_output)

        for class_ in tqdm(listdir(holdout_path)):
            class_path = join(holdout_path, class_)
            if isdir(class_path):
                class_files = [file for file in listdir(class_path) if _is_c_family_file(join(class_path, file))]
                class_output = join(holdout_output, class_)
                class_cpg = join(cpg_path, class_)

                if not exists(class_output):
                    mkdir(class_output)

                if not exists(class_cpg):
                    mkdir(class_cpg)

                class_files_tqdm = tqdm(class_files)

                _ = Parallel(n_jobs=num_cores)(
                    delayed(run_joern_parse)(
                        file=file,
                        class_path=class_path,
                        class_cpg=class_cpg
                    ) for file in class_files_tqdm)

        subprocess.check_call([
            "joern",
            "--script", "preprocess/joern/build_graphs.sc",
            "--params", f"inputPath={holdout_path},cpgPath={cpg_path},outputPath={holdout_output}"
        ])


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config_path", type=str)
    args = arg_parser.parse_args()
    config_ = OmegaConf.load(args.config_path)
    process_graphs(config=config_)
