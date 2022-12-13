import pprint
import random
import numpy as np
import matplotlib.pyplot as plt

from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from libs.files import *
from libs.contents import *
from libs.algorithms import *
from libs.d2v import *
from pqgrams_wrapper import *

import pickle


GOLD_JSON_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2022_ast-path_v0.0.2/json/gold"
GOLD_AST_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2022_ast-path_v0.0.2/ast/gold"
GITHUB_AST_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2022_ast-path_v0.0.2/ast/github"
GOLD_MODEL_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2022_ast-path_v0.0.2/model/gold"
GITHUB_MODEL_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2022_ast-path_v0.0.2/model/github"

VIMAGICK_DATA_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2020_doc2vec_v0.0.1/data/vimagick"
GITHUB_DATA_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2020_doc2vec_v0.0.1/data/github"
VIMAGICK_AST_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2022_ast-path_v0.0.2/ast/vimagick"

def _create_model():
    file_paths = JsonFile._get_file_paths(GITHUB_AST_ROOT_PATH)

    documents = list()
    for file_path in file_paths:
        print(file_path)
        file_basename = JsonFile._get_basename(file_path)
        contents = JsonFile._get_contents(file_path)
        for content in contents:
            astCommand=content["astCommand"]; astCommandId=content["astCommandId"]
            astCommandSequence = Root._get(astCommand)
            documents.append(D2V_ROOT._create_document(tags=astCommandId, sequence=astCommandSequence))

    model_path = "{}/root-pvdm-dim_10.model".format(GITHUB_MODEL_ROOT_PATH)
    D2V_ROOT._create_model(documents=documents, model_path=model_path)


def _confirm_testCaseAst_exists_in_embedding_vector():
    """
    テストケースのASTがLSTMで利用する隠れ層の重みベクトル(Doc2Vecモデル)の中に存在するかを確認, 検証する関数
    """
    test_case = list()
    embedding_case = list()

    file_paths = JsonFile._get_file_paths(GITHUB_AST_ROOT_PATH)
    for file_path in file_paths:
        contents = JsonFile._get_contents(file_path)
        for content in contents:
            dumped = json.dumps(AstCleaner._sort_by_asc(content["astCommand"]))
            if not dumped in embedding_case:
                embedding_case.append(dumped)
    
    count_a = 0
    count_b = 0
    file_paths = JsonFile._get_file_paths(VIMAGICK_AST_ROOT_PATH)
    for file_path in file_paths:
        contents = JsonFile._get_contents(file_path)
        for content in contents:
            dumped = json.dumps(AstCleaner._sort_by_asc(content["astCommand"]))
            count_b += 1
            if not dumped in embedding_case:
                count_a += 1
    print("count_a:", count_a)
    print("count_b:", count_b)
    print("average:", count_a/count_b)
    """
    Result
        GITHUBデータセットはGOLDセットに内包されている(当たり前)

        VIMAGICKデータセット
        count_a: 1164
        count_b: 1264
        average: 0.9208860759493671
    """

def _create_pickle_file_for_time_reduction_of_loading_large_scale_list():
    file_paths = JsonFile._get_file_paths(GITHUB_AST_ROOT_PATH)
    large_scale_list = list()
    for file_path in file_paths:
        contents = JsonFile._get_contents(file_path)
        for content in contents:
            if not content in large_scale_list:
                large_scale_list.append(content)
    
    with open("no_duplication_github_ast_large_collection.bin", mode="wb") as p:
        pickle.dump(large_scale_list, p)
    



def main():
    pass
    # doc2vec2pq()
    # pq2doc2vec()
    # PerFile()
    _create_model()
    # pq2doc2vec_vimagick()
    # _confirm_testCaseAst_exists_in_embedding_vector()
    # _create_pickle_file_for_time_reduction_of_loading_large_scale_list()

if __name__ == "__main__":
    main()