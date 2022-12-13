import pprint
import random
import numpy as np
import matplotlib.pyplot as plt
import json

from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from sklearn.metrics import confusion_matrix


from libs.files import *
from libs.contents import *
from libs.algorithms import *
from libs.d2v import *
from pqgrams_wrapper import *


GOLD_JSON_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2022_ast-path_v0.0.2/json/gold"
GOLD_AST_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2022_ast-path_v0.0.2/ast/gold"
GOLD_MODEL_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2022_ast-path_v0.0.2/model/gold"

VIMAGICK_DATA_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2020_doc2vec_v0.0.1/data/vimagick"
VIMAGICK_AST_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2022_ast-path_v0.0.2/ast/vimagick"

def _create_sampling_case():
    file_paths = JsonFile._get_file_paths(GOLD_AST_ROOT_PATH)

    documents = list()
    for file_path in file_paths:
        documents.extend(JsonFile._get_contents(file_path))

    
    file_path = "/Users/nakamurahekikai/Desktop/binnacle-icse2022_ast-path_v0.0.2/sampling/gold/case/version0.0.5.json"
    JsonFile._create_file(file_path, random.sample(documents, 50))

def _create_sampling_answer():
    file_paths = JsonFile._get_file_paths(VIMAGICK_AST_ROOT_PATH)
    test_datas = list()
    for file_path in file_paths:
        test_datas.extend(JsonFile._get_contents(file_path))
    contents = JsonFile._get_contents("/Users/nakamurahekikai/Desktop/binnacle-icse2022_ast-path_v0.0.2/sampling/vimagick/case/version0.0.1.json")
    documents = list()
    for content in contents:
        results = list()
        for test_data in test_datas:
            pq_edit_distance = PQ_GramWrapper._get_pq_edit_distance(AstCleaner._sort_by_asc(content["astCommand"]), AstCleaner._sort_by_asc(test_data["astCommand"]), P=6, Q=2)
            result = {
                test_data["astCommandId"]: pq_edit_distance
            }
            results.append(result)
        
        document = {
            "results": results,
            "test_case": content["astCommandId"]
        }
        documents.append(document)
    answer_file_path = "/Users/nakamurahekikai/Desktop/binnacle-icse2022_ast-path_v0.0.2/sampling/vimagick/answer/version0.0.1.json"
    JsonFile._create_file(answer_file_path, documents)
        


def main():
    _create_sampling_case()

    # _create_sampling_answer()


    # file_paths = JsonFile._get_file_paths(VIMAGICK_AST_ROOT_PATH)
    # test_datas = list()
    # for file_path in file_paths:
    #     test_datas.extend(JsonFile._get_contents(file_path))


    # contents = JsonFile._get_contents("/Users/nakamurahekikai/Desktop/binnacle-icse2022_ast-path_v0.0.2/sampling/vimagick/case/version0.0.1.json")
    # documents = list()
    # for content in contents:
    #     document = {
    #         "dumped": json.dumps(AstCleaner._sort_by_asc(content["astCommand"])),
    #         "astCommand": AstCleaner._sort_by_asc(content["astCommand"]),
    #         "astCommandId": content["astCommandId"]
    #     }
    #     documents.append(document)
    
    # documents = sorted(documents, key=lambda x:x["dumped"])
    # for document in documents:
    #     pprint.pprint(document["astCommand"])
    #     for test_data in test_datas:
    #         pq_edit_distance = PQ_GramWrapper._get_pq_edit_distance(document["astCommand"], AstCleaner._sort_by_asc(test_data["astCommand"]), P=6, Q=2)
    #         result = {
    #             test_data["astCommandId"]: pq_edit_distance
    #         }
    #         print(result)



if __name__ == "__main__":
    main()