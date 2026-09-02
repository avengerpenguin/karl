import re
import glob
import os
import subprocess
import asyncio
from deepagents.backends import BackendProtocol
from deepagents.backends.protocol import LsResult, ReadResult, GlobResult, WriteResult, GrepResult, FileInfo, GrepMatch, \
    EditResult, FileData, FileDownloadResponse


class ObsidianBackend(BackendProtocol):
    def __init__(self, vault: str):
        self.vault = vault
        self.obsidian_args = [f"vault={self.vault}"]

    def _ensure_parent_folders(self, file_path: str):
        """Helper to ensure subfolders exist using the obsidian CLI before writing."""
        normalized = file_path.lstrip("/")
        parts = normalized.split("/")
        if len(parts) > 1:
            # Reconstruct folder paths incrementally
            for i in range(1, len(parts)):
                folder_path = "/".join(parts[:i])
                # Ensure the folder structure is initiated
                self._cli(["files", f"folder={folder_path}"])

    def ls(self, path: str) -> LsResult:
        path = path.lstrip("/")
        try:
            return LsResult(entries=[
                FileInfo(path="/" + path)
                for path in self._cli(["files", f"folder={path}"])
            ])
        except Exception as e:
            return LsResult(error=str(e))

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        file_path = file_path.lstrip("/")
        try:
            file_lines = self._cli(["read", f"path={file_path}"])
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
            command = ["files", f"folder={path}"] if path else ["files"]
            all_files = self._cli(command)
            re_pattern = re.compile(pattern)
            return GrepResult(matches=[
                GrepMatch(path="/" + file, line=line_num, text=line)
                for file in all_files
                for line_num, line in enumerate(self._cli(["read", f"path={file}"]), start=0)
                if re_pattern.search(line)
            ])
        except Exception as e:
            return GrepResult(error=str(e))

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        path = path.lstrip("/")
        try:
            pattern = re.compile(glob.translate(pattern))
            all_files = self._cli(["files", f"folder={path}"])
            matching_files = [FileInfo(path="/" + file) for file in all_files if pattern.match(file)]
            return GlobResult(matches=matching_files)
        except Exception as e:
            return GlobResult(error=str(e))

    def write(self, file_path: str, content: str) -> WriteResult:
        file_path = file_path.lstrip("/")
        try:
            # self._ensure_parent_folders(file_path)
            all_files = self._cli(["files", file_path])
            if file_path in all_files:
                return WriteResult(error="File exists")
            if "/" in file_path:
                folder_path, file_name = file_path.rsplit("/", 1)
            else:
                folder_path, file_name = "", file_path
            result = self._cli(["create", f"name={file_name}", f"path={folder_path}", f"content={content}"], timeout_seconds=60)
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
            old_content = "\n".join(self._cli(["read", f"path={file_path}"]))
            if old_content.startswith("Error:") and old_content.endswith("not found."):
                return self.write(file_path, new_string)

            occurrences = old_content.count(old_string) if replace_all else 1

            new_content = old_content.replace(
                old_string,
                new_string,
                occurrences if not replace_all else -1,
            )

            self._cli(["create", f"name={file_path}", "overwrite", f"content={new_content}"], timeout_seconds=60)
            return EditResult(path="/" + file_path, occurrences=occurrences)

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

    def _cli(self, args: list[str], timeout_seconds: int = 30) -> list[str]:
        try:
            result = subprocess.run(
                ["obsidian", *self.obsidian_args, *args],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Obsidian CLI timed out after {timeout_seconds}s. "
                f"Command args: {args!r}"
            ) from exc

        if result.returncode != 0:
            raise RuntimeError(
                "Obsidian CLI failed\n"
                f"Exit code: {result.returncode}\n"
                f"Args: {args!r}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        if result.stdout.startswith("Error"):
            print(f"WARNING: Error running {args!r} : {result.stdout}\n{result.stderr}")

        return result.stdout.splitlines()
