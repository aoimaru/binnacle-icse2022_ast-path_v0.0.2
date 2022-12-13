from abc import *
import os

RUN_CLEANING_WORDS = [
    "DOCKER-RUN",
    "BASH-SCRIPT",
    "BASH-AND-IF",
    "BASH-AND-MEM",
    "UNKNOWN"
]

class AstContent(metaclass=ABCMeta):
    pass

class DockerfileAst(AstContent):
    def __init__(self, content):
        self._content = content
    def _get_runs_by_ast(self) -> [dict()]:
        runs_by_ast = list()
        for cnt in self._content:
            if cnt["type"] != "DOCKER-RUN":
                continue
            runs_by_ast.append(cnt)
        return runs_by_ast


class DockerfileRunAst(AstContent):
    def __init__(self, content):
        self._content = content
    def _get_commands_by_ast(self) -> [dict()]:
        commands_by_ast = list()
        def _execute(me):
            if me["children"]:
                for child in me["children"]:
                    if child["type"] in RUN_CLEANING_WORDS:
                        _execute(child)
                    else:
                        commands_by_ast.append(child)
        _execute(self._content)
        return commands_by_ast
    
    def _get_commands_by_ast_with_bash_filter(self):
        commands_by_ast = list()
        def _execute(me):
            if me["children"]:
                for child in me["children"]:
                    if not child["type"].startswith("SC-"):
                        _execute(child)
                    else:
                        commands_by_ast.append(child)
        _execute(self._content)
        return commands_by_ast

class DockerfileCommandsAst(AstContent):
    def __init__(self, content):
        self._content = content
    


class AstCleaner(AstContent):
    @staticmethod
    def _sort_by_asc(content):
        def _execute(me):
            if me["children"]:
                children = list()
                for child in me["children"]:
                    children.append(_execute(child))
                children = sorted(children, key=lambda x:x["type"])
                return {
                    "type": me["type"],
                    "children": children
                }
            else:
                return me

        return _execute(content)