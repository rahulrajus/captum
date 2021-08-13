from abc import ABC, abstractmethod
from captum.compare.fb._core.ModuleSimilarity import ModuleSimilarity
from flask import Flask, send_from_directory
from pathlib import Path
import os.path
from typing import Any

class SimilarityExplorer(ABC):
    def __init__(self, service: ModuleSimilarity, **kwargs):
        self.service = service

        this_filepath = Path(os.path.abspath(__file__))
        this_dirpath = this_filepath.parent.parent
        self.app = Flask(__name__, static_folder=str(this_dirpath.joinpath("frontend", "build")))

        def _serve(subpath="index.html"):
            return send_from_directory(self.app.static_folder, subpath)
        self.app.add_url_rule("/", view_func=_serve)
        self.app.add_url_rule("/<path:subpath>", view_func=_serve)

    @abstractmethod
    def generate_comparison(self, **kwargs: Any):
        pass
    def start(self, debug: bool = True):
        self.app.run(debug=debug)
