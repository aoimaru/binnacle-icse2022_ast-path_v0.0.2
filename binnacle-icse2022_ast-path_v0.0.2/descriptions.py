import pprint
import random
import numpy as np
import matplotlib.pyplot as plt
import json

from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score


from libs.files import *
from libs.contents import *
from libs.algorithms import *
from libs.d2v import *
from pqgrams_wrapper import *


GOLD_JSON_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2022_ast-path_v0.0.2/json/gold"
GOLD_AST_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2022_ast-path_v0.0.2/ast/gold"
GITHUB_AST_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2022_ast-path_v0.0.2/ast/github"
GOLD_MODEL_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2022_ast-path_v0.0.2/model/gold"
GITHUB_MODEL_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2022_ast-path_v0.0.2/model/github"

VIMAGICK_DATA_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2020_doc2vec_v0.0.1/data/vimagick"
VIMAGICK_AST_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2022_ast-path_v0.0.2/ast/vimagick"

PNG_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2022_ast-path_v0.0.2/images"
PNG_DIM_10_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2022_ast-path_v0.0.2/images_dim_10"

class Color(object):
	BLACK          = '\033[30m'#(文字)黒
	RED            = '\033[31m'#(文字)赤
	GREEN          = '\033[32m'#(文字)緑
	YELLOW         = '\033[33m'#(文字)黄
	BLUE           = '\033[34m'#(文字)青
	MAGENTA        = '\033[35m'#(文字)マゼンタ
	CYAN           = '\033[36m'#(文字)シアン
	WHITE          = '\033[37m'#(文字)白
	COLOR_DEFAULT  = '\033[39m'#文字色をデフォルトに戻す
	BOLD           = '\033[1m'#太字
	UNDERLINE      = '\033[4m'#下線
	INVISIBLE      = '\033[08m'#不可視
	REVERCE        = '\033[07m'#文字色と背景色を反転
	BG_BLACK       = '\033[40m'#(背景)黒
	BG_RED         = '\033[41m'#(背景)赤
	BG_GREEN       = '\033[42m'#(背景)緑
	BG_YELLOW      = '\033[43m'#(背景)黄
	BG_BLUE        = '\033[44m'#(背景)青
	BG_MAGENTA     = '\033[45m'#(背景)マゼンタ
	BG_CYAN        = '\033[46m'#(背景)シアン
	BG_WHITE       = '\033[47m'#(背景)白
	BG_DEFAULT     = '\033[49m'#背景色をデフォルトに戻す
	RESET          = '\033[0m'#全てリセット



def test():

    COS_SIM_LIM = 0.80
    PQ_EDIT_LIM = 1.0

    model_path = "{}/root-pvdm.model".format(GITHUB_MODEL_ROOT_PATH)
    # model_path = "{}/root.model".format(GITHUB_MODEL_ROOT_PATH)
    model = D2V_ROOT._load_model(model_path)


    """
    テストケースの評価用のデータを取得
    """
    file_paths = JsonFile._get_file_paths(GOLD_AST_ROOT_PATH)
    test_datas = list()
    for file_path in file_paths:
        test_datas.extend(JsonFile._get_contents(file_path))

    contents = JsonFile._get_contents("/Users/nakamurahekikai/Desktop/binnacle-icse2022_ast-path_v0.0.2/sampling/vimagick/case/version0.0.2.json")
    documents = list()
    for content in contents:
        """
        ASTを辞書順にソートするために, dumpedを定義
        """
        document = {
            "dumped": json.dumps(AstCleaner._sort_by_asc(content["astCommand"])),
            "astCommand": AstCleaner._sort_by_asc(content["astCommand"]),
            "astCommandId": content["astCommandId"]
        }
        documents.append(document)
    """dumpedに基づいて, ASTをソートする"""
    documents = sorted(documents, key=lambda x:x["dumped"])
    for document in documents:

        astCommand = AstCleaner._sort_by_asc(document["astCommand"])
        astCommandSequence = Root._get(astCommand)
        astCommandVector = model.infer_vector(astCommandSequence, epochs=30)

        pprint.pprint(astCommand)
        pq_distances = list()
        cos_sims = list()
        for test_data in test_datas:

            tdAstCommand = AstCleaner._sort_by_asc(test_data["astCommand"])
            tdAstCommandSequence = Root._get(tdAstCommand)
            tdAstCommandVector = model.infer_vector(tdAstCommandSequence, epochs=30)

            pq_edit_distance = PQ_GramWrapper._get_pq_edit_distance(astCommand, tdAstCommand, P=6, Q=2)
            if pq_edit_distance != PQ_EDIT_LIM:
                pq_distances.append(1)
            else:
                pq_distances.append(0)

            cos_sim = D2V._cos_sim(astCommandVector, tdAstCommandVector)
            if cos_sim >= COS_SIM_LIM:
                cos_sims.append(1)
            else:
                cos_sims.append(0)
            # print("result:", "pq_edit_distance:", pq_edit_distance, "cos_sim:", cos_sim)
        pq_distances = np.array(pq_distances)
        cos_sims = np.array(cos_sims)
        try:
            f1 = f1_score(pq_distances, cos_sims)
        except Exception as e:
            f1 = 0.0
        try:
            recall = recall_score(pq_distances, cos_sims)
        except Exception as e:
            recall = 0.0
        try:
            precision = precision_score(pq_distances, cos_sims)
        except Exception as e:
            precision = 0.0
        print(Color.GREEN+"f1_score:{}".format(f1)+Color.RESET, Color.YELLOW+"recall:{}".format(recall)+Color.RESET, Color.BLUE+"precision:{}".format(precision)+Color.RESET)


def count():
    file_paths = JsonFile._get_file_paths(GITHUB_AST_ROOT_PATH)
    print(len(file_paths))
    num_of_contents = 0
    for file_path in file_paths:
        num_of_contents += len(JsonFile._get_contents(file_path))
    print(num_of_contents)
    """
    GITHUBデータセット
    ファイル数: 178452
    コマンド数: 1500682
    """


def view():
    model_path = "{}/root-pvdm-dim_10.model".format(GITHUB_MODEL_ROOT_PATH)
    # model_path = "{}/root.model".format(GITHUB_MODEL_ROOT_PATH)
    model = D2V_ROOT._load_model(model_path)

    
    """
    テストケースの評価用のデータを取得
    """
    file_paths = JsonFile._get_file_paths(GOLD_AST_ROOT_PATH)
    test_datas = list()
    for file_path in file_paths:
        test_datas.extend(JsonFile._get_contents(file_path))

    version = "version0.0.1"
    unit = "gold"
    """テストケースの取得"""
    contents = JsonFile._get_contents("/Users/nakamurahekikai/Desktop/binnacle-icse2022_ast-path_v0.0.2/sampling/{unit}/case/{version}.json".format(unit=unit, version=version))
    documents = list()
    for content in contents:
        """
        ASTを辞書順にソートするために, dumpedを定義
        """
        document = {
            "dumped": json.dumps(AstCleaner._sort_by_asc(content["astCommand"])),
            "astCommand": AstCleaner._sort_by_asc(content["astCommand"]),
            "astCommandId": content["astCommandId"]
        }
        documents.append(document)
    """dumpedに基づいて, ASTをソートする"""
    documents = sorted(documents, key=lambda x:x["dumped"])
    for document in documents:

        astCommand = AstCleaner._sort_by_asc(document["astCommand"])
        astCommandSequence = Root._get(astCommand)
        astCommandVector = model.infer_vector(astCommandSequence, epochs=30)

        pprint.pprint(astCommand)
        pq_edit_distances = list()
        cos_sims = list()
        for test_data in test_datas:

            tdAstCommand = AstCleaner._sort_by_asc(test_data["astCommand"])
            tdAstCommandSequence = Root._get(tdAstCommand)
            tdAstCommandVector = model.infer_vector(tdAstCommandSequence, epochs=30)

            pq_edit_distance = PQ_GramWrapper._get_pq_edit_distance(astCommand, tdAstCommand, P=6, Q=2)
            cos_sim = D2V._cos_sim(astCommandVector, tdAstCommandVector)

            pq_edit_distances.append(pq_edit_distance)
            cos_sims.append(cos_sim)
        png_name = "{root_path}/{unit}/{version}/{name}.png".format(root_path=PNG_DIM_10_PATH, unit=unit, version=version, name=document["astCommandId"])
        cos_sims, pq_edit_distances = png(png_name, document["astCommandId"], cos_sims, pq_edit_distances)

def png(png_name, command_id, cos_sims, pq_edit_distances):
    plt.clf()
    plt.scatter(cos_sims, pq_edit_distances)
    plt.xlabel('cos_dim')
    plt.ylabel('pq_edit_distance')
    plt.title(command_id)
    # plt.show()
    plt.savefig(png_name)
    return list(), list()
    


def main():
    # test()
    # count()
    view()

if __name__ == "__main__":
    main()