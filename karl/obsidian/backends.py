import re
import glob
import subprocess
from deepagents.backends import BackendProtocol
from deepagents.backends.protocol import LsResult, ReadResult, GlobResult, WriteResult, GrepResult, FileInfo, GrepMatch, \
    EditResult, FileData, FileDownloadResponse


class ObsidianBackend(BackendProtocol):
    def __init__(self, vault: str):
        self.vault = vault
        self.command_base = f"obsidian vault='{self.vault}'"

    def ls(self, path: str) -> LsResult:
        path = path.lstrip("/")
        try:
            return LsResult(entries=[
                FileInfo(path="/" + path)
                for path in self._cli(f"files folder='{path}'")
            ])
        except Exception as e:
            return LsResult(error=str(e))

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        file_path = file_path.lstrip("/")
        try:
            file_lines = self._cli(f"read path='{file_path}'")
            if offset > 0:
                file_lines = file_lines[offset:]
            file_lines = file_lines[:limit]
            return ReadResult(file_data=FileData(content='\n'.join(file_lines) + '\n', encoding="utf-8"))
        except Exception as e:
            return ReadResult(error=str(e))

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        try:
            command = f"files 'folder={path}'" if path else "files"
            all_files = self._cli(command)
            re_pattern = re.compile(pattern)
            return GrepResult(matches=[
                GrepMatch(path="/" + file, line=line_num, text=line)
                for file in all_files
                for line_num, line in enumerate(self._cli(f"read 'file={file}'"), start=0)
                if re_pattern.search(line)
            ])
        except Exception as e:
            return GrepResult(error=str(e))

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        path = path.lstrip("/")
        try:
            pattern = re.compile(glob.translate(pattern))
            all_files = self._cli(f"files 'folder={path}'")
            matching_files = [FileInfo(path="/" + file) for file in all_files if pattern.match(file)]
            return GlobResult(matches=matching_files)
        except Exception as e:
            return GlobResult(error=str(e))

    def write(self, file_path: str, content: str) -> WriteResult:
        file_path = file_path.lstrip("/")
        try:
            all_files = self._cli(f"files {file_path}")
            if file_path in all_files:
                return WriteResult(error="File exists")
            new_content = content.replace("'", "'\"'\"'")
            result = self._cli(f"create name='{file_path}' content='{new_content}'")
            return WriteResult(path="/" + file_path)
        except Exception as e:
            return WriteResult(error=str(e))

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        file_path = file_path.lstrip("/")
        try:
            old_content = "\n".join(self._cli(f"read 'file={file_path}'"))
            if replace_all:
                new_content = old_content.replace(old_string, new_string)
            else:
                new_content = old_content.replace(old_string, new_string, count=1)
            new_content = new_content.replace("'", "'\"'\"'")
            self._cli(f"create name='{file_path}' overwrite content='{new_content}'")
            return EditResult(path="/" + file_path)

        except Exception as e:
            return EditResult(error=str(e))

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses: list[FileDownloadResponse] = []

        for path in paths:
            read_result = self.read(path)

            if read_result.error:
                responses.append(FileDownloadResponse(path=path, error=read_result.error))
            else:
                responses.append(
                    FileDownloadResponse(
                        path=path,
                        content=read_result.file_data["content"].encode("utf-8"),
                    )
                )

        return responses

    def _cli(self, command: str) -> list[str]:
        result = subprocess.run(
            f"{self.command_base} {command}", shell=True, capture_output=True, text=True)
        return result.stdout.splitlines()
