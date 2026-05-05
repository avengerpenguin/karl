from typing import AsyncIterator

import os
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_imap import ImapRetriever, ImapConfig

query_prompt = ChatPromptTemplate.from_template(
    """Convert the following user question into IMAP SEARCH criteria.

The generated output will be passed directly to an IMAP SEARCH command.
Therefore, output SEARCH criteria only.

IMAP SEARCH syntax examples:
- 'FROM "john@example.com"' - emails from specific sender
- 'SUBJECT "project update"' - emails with specific subject
- 'SENTSINCE "01-Oct-2024"' - emails since specific date
- 'BODY "meeting"' - emails containing specific word in body
- 'FROM "boss@company.com" SUBJECT "urgent"' - combine criteria
- 'ALL' - all emails in the currently selected mailbox

Folder navigation notes:
- IMAP folders/mailboxes are selected outside SEARCH, for example with SELECT "INBOX" or SELECT "INBOX.jobs".
- Folders underneath INBOX commonly use dotted names, for example INBOX.jobs.
- All folders underneath INBOX can be listed outside SEARCH with LIST "INBOX" "*".
- Do not output SELECT commands.
- Do not output LIST commands.
- Do not include folder names such as INBOX or INBOX.jobs in the SEARCH criteria.
- If the user asks about emails in a folder, generate only the SEARCH criteria for the emails within that folder.
- If the user only asks for emails in a folder and gives no other criteria, output 'ALL'.

IMPORTANT: Include only valid IMAP SEARCH criteria in output.
IMPORTANT: Do not include SELECT, LIST, or any other IMAP command in output.
IMPORTANT: Do not include any other text in output.

User Question: {question}

IMAP Query:"""
)

answer_prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the context provided from emails.

Context:
{context}

Question: {question}

Answer:"""
)

async def check_email(llm: BaseChatModel, messages: list[BaseMessage]) -> AsyncIterator[str]:
    # IMAP retriever configuration
    config = ImapConfig(
        host="mail.rossfenning.co.uk",
        port=993,
        user="post@rossfenning.co.uk",
        password=os.getenv("EMAIL_PASSWORD"),
        ssl_mode="ssl",
        auth_method="login",
        verify_cert=True,
    )

    retriever = ImapRetriever(
        config=config,
        k=5,
        attachment_mode="names_only"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Create the chain
    query_chain = query_prompt | llm | StrOutputParser()

    def generate_imap_query(question):
        return query_chain.invoke({"question": question})

    def search_emails(query):
        return retriever.invoke(query)

    full_chain = (
        {
            "question": lambda x: x,
            "imap_query": lambda x: generate_imap_query(x)
        }
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(search_emails(x["imap_query"]))
        )
        | answer_prompt
        | llm
    )

    async for chunk in full_chain.astream(messages):
        yield chunk
