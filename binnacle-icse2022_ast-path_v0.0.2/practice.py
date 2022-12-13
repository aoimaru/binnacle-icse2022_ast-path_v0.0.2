import pprint

# import numpy as np
import matplotlib.pyplot as plt

from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from libs.files import *
from libs.contents import *
from libs.algorithms import *
from pqgrams_wrapper import *


GOLD_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2022_ast-path_v0.0.2/gold"


def _create_model():
    file_paths = JsonFile._get_file_paths(GOLD_AST_ROOT_PATH)

    documents = list()
    for file_path in file_paths:
        file_basename = JsonFile._get_basename(file_path)
        contents = JsonFile._get_contents(file_path)
        for content in contents:
            astCommand=content["astCommand"]; astCommandId=content["astCommandId"]
            astCommandSequence = Root._get(astCommand)
            documents.append(D2V_ROOT._create_document(tags=astCommandId, sequence=astCommandSequence))
    
    model_path = "{}/root.model".format(GOLD_MODEL_ROOT_PATH)
    D2V_ROOT._create_model(documents=documents, model_path=model_path)

            

# def PerFile():
#     file_paths = JsonFile._get_file_paths(GOLD_JSON_ROOT_PATH)

#     for file_path in file_paths:
#         file_basename = JsonFile._get_basename(file_path)
#         content = JsonFile._get_contents(file_path)
#         dockerfileAst = DockerfileAst(content)
#         astRuns = dockerfileAst._get_runs_by_ast()
#         ast_file_path = "{}/{}.json".format(GOLD_AST_ROOT_PATH, file_basename)
#         astCommandsPerFile = list()
#         for astRunId, astRun in enumerate(astRuns):
#             dockerfileRunAst = DockerfileRunAst(astRun)
#             astCommands = dockerfileRunAst._get_commands_by_ast_with_bash_filter()
#             for astCommandId, astCommand in enumerate(astCommands):
#                 astCommand = AstCleaner._sort_by_asc(astCommand)
#                 astCommandId = "{basename}:{run_id}:{command_id}".format(basename=file_basename, run_id=astRunId, command_id=astCommandId)
#                 astCommandPer = {
#                     "astCommandId": astCommandId,
#                     "astCommand": astCommand
#                 }
#                 astCommandsPerFile.append(astCommandPer)
        
#         JsonFile._create_file(ast_file_path, astCommandsPerFile)

def PerFile():
    file_paths = JsonFile._get_file_paths(VIMAGICK_DATA_ROOT_PATH)
    VIMAGICK_AST_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2022_ast-path_v0.0.2/ast/vimagick"
    for file_path in file_paths:
        file_basename = JsonFile._get_basename(file_path)
        content = JsonFile._get_contents(file_path)
        dockerfileAst = DockerfileAst(content)
        astRuns = dockerfileAst._get_runs_by_ast()
        ast_file_path = "{}/{}.json".format(VIMAGICK_AST_ROOT_PATH, file_basename)
        astCommandsPerFile = list()
        for astRunId, astRun in enumerate(astRuns):
            dockerfileRunAst = DockerfileRunAst(astRun)
            astCommands = dockerfileRunAst._get_commands_by_ast_with_bash_filter()
            for astCommandId, astCommand in enumerate(astCommands):
                astCommand = AstCleaner._sort_by_asc(astCommand)
                astCommandId = "{basename}:{run_id}:{command_id}".format(basename=file_basename, run_id=astRunId, command_id=astCommandId)
                astCommandPer = {
                    "astCommandId": astCommandId,
                    "astCommand": astCommand
                }
                astCommandsPerFile.append(astCommandPer)
        
        JsonFile._create_file(ast_file_path, astCommandsPerFile)

def doc2vec2pq():
    model_path = "{}/root-pvdm.model".format(GOLD_MODEL_ROOT_PATH)
    # model_path = "{}/root.model".format(GOLD_MODEL_ROOT_PATH)
    model = D2V_ROOT._load_model(model_path)
    file_paths = JsonFile._get_file_paths(GOLD_AST_ROOT_PATH)

    test_data = list()
    for file_path in file_paths:
        file_basename = JsonFile._get_basename(file_path)
        test_data.extend(JsonFile._get_contents(file_path))


    for content in random.sample(test_data, 5):
        astCommand=content["astCommand"]; astCommandId=content["astCommandId"]
        astCommand = AstCleaner._sort_by_asc(astCommand)
        print()
        print()
        print()
        # pprint.pprint(astCommand)
        for most_sim in model.docvecs.most_similar(astCommandId)[:20]:
            # print("score           :", most_sim[1])
            for td in test_data:
                if td["astCommandId"]==most_sim[0]:
                    # pprint.pprint(td)
                    pq_edit_distance = PQ_GramWrapper._get_pq_edit_distance(astCommand, td["astCommand"], P=6, Q=2)
                    print("score:", most_sim[1], "pq_edit_distance:", pq_edit_distance)
                    break

def pq2doc2vec():
    model_path = "{}/root-pvdm.model".format(GOLD_MODEL_ROOT_PATH)
    # model_path = "{}/root.model".format(GOLD_MODEL_ROOT_PATH)
    model = D2V_ROOT._load_model(model_path)
    file_paths = JsonFile._get_file_paths(GOLD_AST_ROOT_PATH)
    test_data = list()
    for file_path in file_paths:
        file_basename = JsonFile._get_basename(file_path)
        test_data.extend(JsonFile._get_contents(file_path))


    for content in random.sample(test_data, 5):
        astCommand=content["astCommand"]; astCommandId=content["astCommandId"]
        astCommand = AstCleaner._sort_by_asc(astCommand)
        pprint.pprint(astCommand)
        results = list()
        for td in test_data:
            tdAstCommand = AstCleaner._sort_by_asc(td["astCommand"])
            pq_edit_distance = PQ_GramWrapper._get_pq_edit_distance(astCommand, tdAstCommand, P=6, Q=2)
            result = {
                "tdAstCommand": tdAstCommand,
                "pq_edit_distance": pq_edit_distance,
                "tdAstCommandId": td["astCommandId"]
            }
            results.append(result)
        results = sorted(results, key=lambda x:x["pq_edit_distance"])

        hg = list()
        wd = list()
        for result in results:
            cos_sim = model.docvecs.similarity(astCommandId, result["tdAstCommandId"])
            # print(result["pq_edit_distance"], cos_sim)
            # pprint.pprint(result["tdAstCommand"])
            hg.append(result["pq_edit_distance"])
            wd.append(cos_sim)
        
        plt.scatter(wd, hg)
        plt.xlabel('cos_dim')
        plt.ylabel('pq_edit_distance')
        plt.show()

def pq2doc2vec_vimagick():
    model_path = "{}/root-pvdm.model".format(GOLD_MODEL_ROOT_PATH)
    model_path = "{}/root.model".format(GOLD_MODEL_ROOT_PATH)
    model = D2V_ROOT._load_model(model_path)
    VIMAGICK_AST_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2022_ast-path_v0.0.2/ast/vimagick"
    file_paths = JsonFile._get_file_paths(VIMAGICK_AST_ROOT_PATH)
    test_data = list()
    for file_path in file_paths:
        file_basename = JsonFile._get_basename(file_path)
        test_data.extend(JsonFile._get_contents(file_path))

    for content in random.sample(test_data, 3):
        astCommand=content["astCommand"]; astCommandId=content["astCommandId"]
        astCommand = AstCleaner._sort_by_asc(astCommand)
        astCommandSequence = Root._get(astCommand)
        astCommandVector = model.infer_vector(astCommandSequence, epochs=30)
        pprint.pprint(astCommand)
        results = list()
        for td in test_data:
            tdAstCommand = AstCleaner._sort_by_asc(td["astCommand"])
            pq_edit_distance = PQ_GramWrapper._get_pq_edit_distance(astCommand, tdAstCommand, P=6, Q=2)
            result = {
                "tdAstCommand": tdAstCommand,
                "pq_edit_distance": pq_edit_distance,
                "tdAstCommandId": td["astCommandId"]
            }
            results.append(result)
        results = sorted(results, key=lambda x:x["pq_edit_distance"])

        hg = list()
        wd = list()
        for result in results:
            try:
                pqAstCommandSequence = Root._get(result["tdAstCommand"])
                pqAstCommandVector = model.infer_vector(pqAstCommandSequence, epochs=30)
                cos_sim = D2V._cos_sim(astCommandVector, pqAstCommandVector)
                # print(result["tdAstCommand"])
                hg.append(result["pq_edit_distance"])
                wd.append(cos_sim)
            except Exception as e:
                print(e)
        
        plt.scatter(wd, hg)
        plt.xlabel('cos_dim')
        plt.ylabel('pq_edit_distance')
        plt.show()

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
    
    model_path = "{}/root.model".format(GITHUB_MODEL_ROOT_PATH)
    D2V_ROOT._create_model(documents=documents, model_path=model_path)


def PerFile():
    file_paths = JsonFile._get_file_paths(GITHUB_DATA_ROOT_PATH)
    GITHUB_AST_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2022_ast-path_v0.0.2/ast/github"
    for file_path in file_paths:
        file_basename = JsonFile._get_basename(file_path)
        content = JsonFile._get_contents(file_path)
        dockerfileAst = DockerfileAst(content)
        astRuns = dockerfileAst._get_runs_by_ast()
        ast_file_path = "{}/{}.json".format(GITHUB_AST_ROOT_PATH, file_basename)
        astCommandsPerFile = list()
        for astRunId, astRun in enumerate(astRuns):
            dockerfileRunAst = DockerfileRunAst(astRun)
            astCommands = dockerfileRunAst._get_commands_by_ast_with_bash_filter()
            for astCommandId, astCommand in enumerate(astCommands):
                astCommand = AstCleaner._sort_by_asc(astCommand)
                astCommandId = "{basename}:{run_id}:{command_id}".format(basename=file_basename, run_id=astRunId, command_id=astCommandId)
                astCommandPer = {
                    "astCommandId": astCommandId,
                    "astCommand": astCommand
                }
                print(astCommandPer)
                astCommandsPerFile.append(astCommandPer)
        
        JsonFile._create_file(ast_file_path, astCommandsPerFile)




def main():
    file_paths = JsonFile._get_file_paths(GOLD_ROOT_PATH)

    # src_content = JsonFile._get_contents(file_paths[5])
    # src_dockerfileAst = DockerfileAst(src_content)
    # src_astRuns = src_dockerfileAst._get_runs_by_ast()
    # src_astRun = DockerfileRunAst(src_astRuns[3])
    # src_astCommands = src_astRun._get_commands_by_ast_with_bash_filter()
    # src_astCommand = AstCleaner._sort_by_asc(src_astCommands[1])
    # pprint.pprint(src_astCommand)
    # astCommandPath = Root._get(src_astCommand)
    # print(astCommandPath)


    # # pprint.pprint(src_astCommand)
    # src_content = JsonFile._get_contents(file_paths[10])
    # src_dockerfileAst = DockerfileAst(src_content)
    # src_astRuns = src_dockerfileAst._get_runs_by_ast()
    # src_astRun = DockerfileRunAst(src_astRuns[3])
    # src_astCommands = src_astRun._get_commands_by_ast_with_bash_filter()
    # src_astCommand = AstCleaner._sort_by_asc(src_astCommands[1])
    # # pprint.pprint(src_astCommand)
    # src_content = JsonFile._get_contents(file_paths[30])
    # src_dockerfileAst = DockerfileAst(src_content)
    # src_astRuns = src_dockerfileAst._get_runs_by_ast()
    # src_astRun = DockerfileRunAst(src_astRuns[2])
    # src_astCommands = src_astRun._get_commands_by_ast_with_bash_filter()
    # src_astCommand = AstCleaner._sort_by_asc(src_astCommands[1])
    # # pprint.pprint(src_astCommand)
    # src_content = JsonFile._get_contents(file_paths[33])
    # src_dockerfileAst = DockerfileAst(src_content)
    # src_astRuns = src_dockerfileAst._get_runs_by_ast()
    # src_astRun = DockerfileRunAst(src_astRuns[2])
    # src_astCommands = src_astRun._get_commands_by_ast_with_bash_filter()
    # src_astCommand = AstCleaner._sort_by_asc(src_astCommands[1])
    # pprint.pprint(src_astCommand)
    # astCommandPath = Root._get(src_astCommand)
    # print(astCommandPath)

    # src_content = JsonFile._get_contents(file_paths[37])
    # src_dockerfileAst = DockerfileAst(src_content)
    # src_astRuns = src_dockerfileAst._get_runs_by_ast()
    # src_astRun = DockerfileRunAst(src_astRuns[2])
    # src_astCommands = src_astRun._get_commands_by_ast_with_bash_filter()
    # src_astCommand = AstCleaner._sort_by_asc(src_astCommands[1])
    # pprint.pprint(src_astCommand)
    # astCommandPath = Root._get(src_astCommand)
    # print(astCommandPath)
    # src_content = JsonFile._get_contents(file_paths[47])
    # src_dockerfileAst = DockerfileAst(src_content)
    # src_astRuns = src_dockerfileAst._get_runs_by_ast()
    # src_astRun = DockerfileRunAst(src_astRuns[2])
    # src_astCommands = src_astRun._get_commands_by_ast_with_bash_filter()
    # src_astCommand = AstCleaner._sort_by_asc(src_astCommands[1])
    # # pprint.pprint(src_astCommand)
    # # Root._get(src_astCommand)
    src_content = JsonFile._get_contents(file_paths[67])
    src_dockerfileAst = DockerfileAst(src_content)
    src_astRuns = src_dockerfileAst._get_runs_by_ast()
    src_astRun = DockerfileRunAst(src_astRuns[2])
    src_astCommands = src_astRun._get_commands_by_ast_with_bash_filter()
    src_astCommand = AstCleaner._sort_by_asc(src_astCommands[1])
    pprint.pprint(src_astCommand)
    astCommandPath = Root._get(src_astCommand)
    print(astCommandPath)

    results = list()

    for file_path in file_paths:
        content = JsonFile._get_contents(file_path)
        dockerfileAst = DockerfileAst(content)
        astRuns = dockerfileAst._get_runs_by_ast()
        for astRun in astRuns:
            dockerfileRunAst = DockerfileRunAst(astRun)
            astCommands = dockerfileRunAst._get_commands_by_ast_with_bash_filter()
            for astCommand in astCommands:
                astCommand = AstCleaner._sort_by_asc(astCommand)
                # astCommandPath = Root._get(astCommand)
                # print(astCommandPath)
                pq_edit_distance = PQ_GramWrapper._get_pq_edit_distance(src_astCommand, astCommand, P=6, Q=2)
                result = {
                    "astCommand": astCommand,
                    "pq_edit_distance": pq_edit_distance
                }
                results.append(result)
    
    print()
    print()
    # results = sorted(results, key=lambda x:x["pq_edit_distance"])
    # results = [result for result in results if 0.9 <= result["pq_edit_distance"] <= 0.93]
    # for result in results[:10]:
    #     pprint.pprint(result)



    pq_edit_distances = dict()
    for result in results:
        pq_edit_distance = str(result["pq_edit_distance"])[:4]
        if not pq_edit_distance in pq_edit_distances:
            pq_edit_distances[pq_edit_distance] = 0
        pq_edit_distances[pq_edit_distance] += 1
    
    pprint.pprint(pq_edit_distances)
    
    # plt.bar(left, height)
    # plt.show()
    
if __name__ == "__main__":
    main()