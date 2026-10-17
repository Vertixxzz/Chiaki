from pathlib import Path


class AnalyzerNotConfiguredError(RuntimeError):
    pass


class Analyzer:
    def analyze(self, video_path: Path) -> dict:
        raise AnalyzerNotConfiguredError(
            "analyzer_not_configured"
        )