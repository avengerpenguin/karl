"""Imap retrievers."""

import imaplib
import os
import ssl
import tempfile
from contextlib import contextmanager
from email import message_from_bytes
from email.header import decode_header
from email.message import EmailMessage
from email.policy import default
from io import BytesIO
from typing import Any, Iterator, List, Literal, Union

from html_to_markdown import convert_to_markdown
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pypdf import PdfReader

from langchain_imap.utils import ImapConfig


class ImapRetriever(BaseRetriever):
    """IMAP retriever to search and retrieve emails as documents.

    This retriever connects to an IMAP server, searches for emails based on IMAP query
    syntax, and returns matching emails formatted as LangChain documents.

    Setup:
        Install ``langchain-imap`` and optionally extras for attachment processing:

        .. code-block:: bash

            pip install langchain-imap
            # For basic text extraction from attachments:
            pip install langchain-imap[text_extract]
            # For full document processing (PDF, DOCX, etc):
            pip install langchain-imap[full_content]

    Key init args:
        config: ImapConfig
            Configuration dictionary containing IMAP connection details:
            - 'host' (str): IMAP server hostname (e.g., 'imap.gmail.com')
            - 'port' (int): IMAP server port (e.g., 993 for SSL)
            - 'user' (str): Username/email for authentication
            - 'password' (str): User's password for 'login' auth,
              or the access token for 'auth_xoauth'.
            - 'auth_method' (str, optional): Authentication method. Can be "login",
              "auth_cram_md5", "auth_plain", or "auth_xoauth". Defaults to "login".
            - 'ssl_mode' (str, optional): SSL mode: 'plain' for plain,
              'ssl' for direct SSL, 'starttls' for STARTTLS. Defaults to "ssl".
        attachment_mode: str, default 'names_only'
            How to handle email attachments:
            - 'names_only': Include only attachment filenames in document
            - 'text_extract': Extract text from text/plain and PDF attachments
            - 'full_content': Use comprehensive document processing for supported
              formats using docling (require docling).
        include_attachments_in_content: bool, default True
            Whether to include processed attachment content in the document page_content

    Instantiate:
        .. code-block:: python

            from langchain_imap import ImapRetriever

            config = {
                "host": "imap.gmail.com",
                "port": 993,
                "user": "your-email@gmail.com",
                "password": "your-app-password"
            }
            retriever = ImapRetriever(
                config=ImapConfig(**config),
                attachment_mode="text_extract"
            )

    Usage:
        .. code-block:: python

            # Search for emails from specific sender with subject
            query = 'FROM john@example.com SUBJECT "project update"'
            results = retriever.invoke(query)

            for doc in results:
                print(doc.page_content)

        .. code-block:: none

            # Example output (simplified):
            To: you@example.com
            From: john@example.com
            Subject: Project Update - Q3 Results
            CC:
            BCC:
            Date: 2024-01-15 10:30:00
            Body: Here are the Q3 results...
            Attachments: results.pdf

    Use within a chain:
        .. code-block:: python

            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnablePassthrough
            from langchain_openai import ChatOpenAI

            prompt = ChatPromptTemplate.from_template(
                \"""Answer the question based only on the context provided.

            Context: {context}

            Question: {question}\""
            )

            llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

            def format_docs(docs):
                return "\\n\\n".join(doc.page_content for doc in docs)

            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            result = chain.invoke(
                "What are the key points from John's email about Q3 results?"
            )

        .. code-block:: none

            # Output would be an AI-generated summary based on retrieved email content.

    """

    config: ImapConfig
    """IMAP configuration."""
    k: int = 3
    """Number of results to fetch."""
    attachment_mode: Literal["names_only", "text_extract", "full_content"] = (
        "names_only"
    )
    """Mode for handling attachments."""
    include_attachments_in_content: bool = True
    """Whether to include attachment content in the document."""
    timeout: int = 30
    """Timeout in seconds for the connection attempt"""

    @contextmanager
    def _get_mailbox(self) -> Iterator[Union[imaplib.IMAP4, imaplib.IMAP4_SSL]]:
        """Create and manage IMAP connection using imaplib."""
        ssl_context = None
        if self.config.ssl_mode != "plain" and self.config.verify_cert:
            ssl_context = ssl.create_default_context(
                cafile=self.config.cafile, capath=self.config.capath
            )

        # Create connection based on SSL mode
        imap: Union[imaplib.IMAP4, imaplib.IMAP4_SSL]
        if self.config.ssl_mode == "ssl":
            imap = imaplib.IMAP4_SSL(
                self.config.host,
                port=self.config.port,
                ssl_context=ssl_context,
                timeout=self.timeout,
            )
        else:  # "plain" or "starttls"
            imap = imaplib.IMAP4(
                self.config.host, port=self.config.port, timeout=self.timeout
            )

            if self.config.ssl_mode == "starttls":
                imap.starttls(ssl_context)

        try:
            # Authentication
            self._authenticate(imap)

            # Select INBOX (default mailbox)
            imap.select()
            yield imap

        finally:
            try:
                imap.close()
                imap.logout()
            except Exception:
                pass

    def _authenticate(self, imap: Union[imaplib.IMAP4, imaplib.IMAP4_SSL]) -> None:
        """Handle different authentication methods."""

        if self.config.auth_method == "login":
            # Uses IMAP LOGIN command
            imap.login(self.config.user, self.config.password)

        elif self.config.auth_method == "auth_cram_md5":
            # Uses AUTHENTICATE command with CRAM-MD5 mechanism
            imap.login_cram_md5(self.config.user, self.config.password)

        elif self.config.auth_method == "auth_plain":
            # TODO: Implement PLAIN authentication via authenticate()
            # Example: imap.authenticate('PLAIN', self._plain_auth_handler)
            raise NotImplementedError("PLAIN authentication not yet implemented")

        elif self.config.auth_method == "auth_xoauth":
            # TODO: Implement XOAUTH authentication via authenticate()
            # Example: imap.authenticate('XOAUTH', self._xoauth_auth_handler)
            raise NotImplementedError("XOAUTH authentication not yet implemented")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        """Fetch emails using imaplib and convert to documents."""
        k = kwargs.get("k", self.k)
        documents: List[Document] = []

        try:
            with self._get_mailbox() as imap:
                # Search for messages
                typ, msg_data = imap.search(None, query)
                if typ != "OK":
                    return documents

                # Get message IDs with type guard
                if not msg_data or not isinstance(msg_data[0], bytes):
                    return documents
                msg_ids = msg_data[0].split()

                # Limit results
                msg_ids = msg_ids[:k]

                # Fetch each message
                for msg_id in msg_ids:
                    msg_id_str = msg_id.decode()
                    typ, msg_data = imap.fetch(msg_id_str, "(RFC822)")
                    if (
                        typ == "OK"
                        and isinstance(msg_data, list)
                        and isinstance(msg_data[0], tuple)
                    ):
                        raw_email = msg_data[0][1]
                        document = self._parse_email(raw_email, msg_id_str)
                        documents.append(document)
                    else:
                        raise ValueError(
                            f"Failed to fetch message {msg_id_str}"
                            f"return typ is {typ}"
                            f"data type is {type(msg_data)}"
                        )
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve emails: {e}") from e
        return documents

    def _parse_email(self, raw_email: bytes, msg_id: str) -> Document:
        """Parse raw email bytes into Document using EmailMessage."""
        # Use EmailMessage with modern policy for better parsing
        msg = message_from_bytes(raw_email, policy=default)

        # Extract headers using EmailMessage methods
        subject = self._decode_header(str(msg.get("Subject", "")))
        from_addr = self._decode_header(str(msg.get("From", "")))
        to_addr = self._decode_header(str(msg.get("To", "")))
        cc_addr = self._decode_header(str(msg.get("CC", "")))
        bcc_addr = self._decode_header(str(msg.get("BCC", "")))
        date = msg.get("Date", "")

        # Convert date to ISO format for consistency
        if date:
            try:
                from email.utils import parsedate_to_datetime

                dt = parsedate_to_datetime(date)
                if dt:
                    date = dt.isoformat()
            except Exception:
                pass  # Keep original date if parsing fails

        # Extract body using improved method
        body = self._extract_body(msg)

        # Process attachments using improved method
        attachment_info = self._process_attachments_emailmessage(msg)

        # Format content
        page_content = self._format_email_content(
            to_addr, from_addr, subject, cc_addr, bcc_addr, date, body, attachment_info
        )

        return Document(
            page_content=page_content,
            metadata={
                "message_id": msg_id,
                "date": date,
                "from": from_addr,
                "to": to_addr,
                "subject": subject,
            },
        )

    def _decode_header(self, header: str) -> str:
        """Decode email header with proper encoding."""
        decoded_parts = []
        for part, encoding in decode_header(header):
            if isinstance(part, bytes):
                decoded_parts.append(part.decode(encoding or "utf-8", errors="ignore"))
            else:
                decoded_parts.append(part)
        return "".join(decoded_parts)

    def _extract_body(self, msg: EmailMessage) -> str:
        """Extract text body from email message using EmailMessage.get_body()."""
        # Use EmailMessage.get_body() for intelligent body selection
        # Prefer HTML content, fallback to plain text
        body_part = msg.get_body(preferencelist=("html", "plain"))
        if body_part:
            content_type = body_part.get_content_type()
            payload = body_part.get_payload(decode=True)

            if payload and isinstance(payload, bytes):
                charset = body_part.get_content_charset() or "utf-8"
                content = payload.decode(charset, errors="ignore")

                if content_type == "text/html":
                    # Convert HTML to markdown
                    return convert_to_markdown(content, heading_style="atx")
                else:
                    return content

        return ""

    def _process_attachments_emailmessage(self, msg: EmailMessage) -> str:
        """Process attachments using EmailMessage.iter_attachments()."""
        if self.attachment_mode == "names_only":
            return self._get_attachment_names(msg)
        elif self.attachment_mode in ("text_extract", "full_content"):
            return self._extract_attachment_content(msg)
        return ""

    def _get_attachment_names(self, msg: EmailMessage) -> str:
        """Get list of attachment filenames using EmailMessage.iter_attachments()."""
        attachments = ""
        for attachment in msg.iter_attachments():
            filename = attachment.get_filename()
            if filename:
                attachments += f"- `{filename}`\n"

        return attachments

    def _extract_attachment_content(self, msg: EmailMessage) -> str:
        """Extract content from attachments using EmailMessage.iter_attachments()."""
        contents = []
        for attachment in msg.iter_attachments():
            filename = attachment.get_filename()
            if filename:
                content = self._process_attachment_part(attachment)
                if content:
                    contents.append(f"``` `{filename}`\n{content}\n```")

        return "\n\n".join(contents)

    def _process_attachment_part(self, part: EmailMessage) -> str | None:
        """Process a single attachment part."""
        if self.attachment_mode == "text_extract":
            return self._extract_text_simple(part)
        elif self.attachment_mode == "full_content":
            return self._extract_content_full(part)
        return None

    def _extract_text_simple(self, part: EmailMessage) -> str | None:
        """Extract text from simple attachment types using EmailMessage methods."""
        # Use EmailMessage methods for better content type detection
        content_type = part.get_content_type()
        content_disposition = part.get_content_disposition()
        charset = part.get_content_charset() or "utf-8"
        payload = part.get_payload(decode=True)

        if not payload or content_disposition != "attachment":
            return None

        # Ensure payload is bytes before proceeding
        if not isinstance(payload, bytes):
            return None

        # Enhanced content type handling with EmailMessage methods
        if content_type == "text/plain":
            try:
                return payload.decode(charset, errors="ignore")
            except Exception:
                return None
        elif content_type == "application/pdf":
            try:
                pdf = PdfReader(BytesIO(payload))
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                return text.strip() or None
            except Exception:
                return None
        return None

    def _extract_content_full(self, part: EmailMessage) -> str | None:
        """
        Extract content using docling for full document processing with EmailMessage
        methods.
        """
        # Use EmailMessage methods for better content type detection
        content_disposition = part.get_content_disposition()
        payload = part.get_payload(decode=True)

        if not payload or content_disposition != "attachment":
            return None

        # Ensure payload is bytes before proceeding
        if not isinstance(payload, bytes):
            return None

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
            try:
                tmp_file.write(payload)
                tmp_file.close()
                from docling.document_converter import DocumentConverter

                converter = DocumentConverter()
                result = converter.convert(tmp_path)
                return result.document.export_to_markdown()
            except ImportError:
                return None
            except Exception:
                return None
            finally:
                os.unlink(tmp_path)

    def _format_email_content(
        self,
        to_addr: str,
        from_addr: str,
        subject: str,
        cc_addr: str,
        bcc_addr: str,
        date: str,
        body: str,
        attachment_info: str,
    ) -> str:
        """Format email content into a readable string."""
        lines = [
            f"To: {to_addr}",
            f"From: {from_addr}",
            f"Subject: {subject}",
            f"CC: {cc_addr}",
            f"BCC: {bcc_addr}",
            f"Date: {date}",
            f"Body: {body}",
        ]

        if attachment_info:
            lines.append(f"Attachments: {attachment_info}")

        return "\n".join(lines)

    # TODO: Implement future authentication handlers
    # def _plain_auth_handler(self, response):
    #     """Handler for PLAIN authentication."""
    #     # Implementation for PLAIN auth
    #     pass

    # def _xoauth_auth_handler(self, response):
    #     """Handler for XOAUTH authentication."""
    #     # Implementation for XOAUTH auth
    #     pass

    # optional: add custom async implementations here
    # async def _aget_relevant_documents(
    #     self,
    #     query: str,
    #     *,
    #     run_manager: AsyncCallbackManagerForRetrieverRun,
    #     **kwargs: Any,
    # ) -> List[Document]: ...
