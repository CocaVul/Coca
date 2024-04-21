from CppCodeAnalyzer.extraTools.vuldetect.deepwukong import *
from CppCodeAnalyzer.mainTool.CPG import initialCalleeInfos, CFGToUDGConverter, ASTDefUseAnalyzer
from time import time
import sys
import os
import tqdm
import json

from global_defines import cur_dir


def test_vul():
    start = time()
    dataset = f"{cur_dir}/datasets/vulgen/test"
    configuration = f"{cur_dir}/detectors/Reveal/calleeInfos.json"

    calleeInfs = json.load(open(configuration, 'r', encoding='utf-8'))
    calleeInfos = initialCalleeInfos(calleeInfs)

    astAnalyzer: ASTDefUseAnalyzer = ASTDefUseAnalyzer()
    astAnalyzer.calleeInfos = calleeInfos
    converter: CFGToUDGConverter = CFGToUDGConverter()
    converter.astAnalyzer = astAnalyzer
    defUseConverter: CFGAndUDGToDefUseCFG = CFGAndUDGToDefUseCFG()
    ddgCreator: DDGCreator = DDGCreator()

    vul_json_content = []
    nor_json_content = []
    json_vul_path = f"{cur_dir}/datasets/test_vul.json"
    json_nor_path = f"{cur_dir}/datasets/test_nor.json"
    json_vul = open(json_vul_path, mode='w')
    json_nor = open(json_nor_path, mode='w')
    num_vul_file = len(os.listdir(dataset))
    failure = 0

    with tqdm.tqdm(total=num_vul_file, leave=True, ncols=200, unit_scale=False) as bar:
        for root, dirs, files in os.walk(dataset):
            for file in files:
                if file.endswith("_vul.c"):
                    bar.update(1)
                    bar.set_postfix({"Current Processed File": file})
                    path = os.path.join(root, file)
                    try:
                        cpgs: List[CPG] = fileParse(path, converter, defUseConverter, ddgCreator)
                        for cpg in cpgs:
                            # json.dump(cpg.toSerializedJson(), json_file, indent=2)
                            vul_json_content.append(cpg.toSerializedJson())
                    except:
                        failure += 1
                        break
                """
                if file.endswith("_nonvul.c"):
                    bar.update(1)
                    bar.set_postfix({"Current Processed File": file})
                    path = os.path.join(root, file)
                    try:
                        cpgs: List[CPG] = fileParse(path, converter, defUseConverter, ddgCreator)
                        for cpg in cpgs:
                            # json.dump(cpg.toSerializedJson(), json_file, indent=2)
                            nor_json_content.append(cpg.toSerializedJson())
                    except:
                        failure += 1
                        break
                """

    json.dump(vul_json_content, json_vul, indent=2)
    # json.dump(nor_json_content, json_nor, indent=2)

    end = time()
    print(f"Successfully compile {num_vul_file - failure} samples")
    print(f"Fail to compile {failure} samples")
    print('Total time: {:.2f}s'.format(end - start))
    return


def test_nor():
    start = time()
    dataset = f"{cur_dir}/datasets/vulgen/test"
    configuration = f"{cur_dir}/detectors/Reveal/calleeInfos.json"

    calleeInfs = json.load(open(configuration, 'r', encoding='utf-8'))
    calleeInfos = initialCalleeInfos(calleeInfs)

    astAnalyzer: ASTDefUseAnalyzer = ASTDefUseAnalyzer()
    astAnalyzer.calleeInfos = calleeInfos
    converter: CFGToUDGConverter = CFGToUDGConverter()
    converter.astAnalyzer = astAnalyzer
    defUseConverter: CFGAndUDGToDefUseCFG = CFGAndUDGToDefUseCFG()
    ddgCreator: DDGCreator = DDGCreator()

    save_json_content = []
    json_nor_path = f"{cur_dir}/datasets/test_nor.json"
    json_nor = open(json_nor_path, mode='w')
    num_vul_file = len(os.listdir(dataset))
    failure = 0

    with tqdm.tqdm(total=num_vul_file, leave=True, ncols=200, unit_scale=False) as bar:
        for root, dirs, files in os.walk(dataset):
            for file in files:
                if file.endswith("_nonvul.c"):
                    bar.update(1)
                    bar.set_postfix({"Current Processed File": file})
                    path = os.path.join(root, file)
                    try:
                        cpgs: List[CPG] = fileParse(path, converter, defUseConverter, ddgCreator)
                        for cpg in cpgs:
                            # json.dump(cpg.toSerializedJson(), json_file, indent=2)
                            save_json_content.append(cpg.toSerializedJson())
                    except:
                        failure += 1
                        break

    json.dump(save_json_content, json_nor, indent=2)

    end = time()
    print(f"Successfully compile {failure} samples")
    print(f"Fail to compile {num_vul_file - failure} samples")
    print('Total time: {:.2f}s'.format(end - start))
    return


if __name__ == '__main__':
    test_vul()
    # test_nor()
