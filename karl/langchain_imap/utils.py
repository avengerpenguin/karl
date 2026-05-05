"""IMAP utilities."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ImapConfig(BaseModel):
    """IMAP configuration model.
    
    :attribute host: IMAP server hostname (e.g., 'imap.gmail.com')
    """

    host: str = Field(
        description="IMAP server hostname (e.g., 'imap.gmail.com')", min_length=1
    )
    user: str = Field(description="Username/email for authentication", min_length=1)
    password: str = Field(
        description=(
            "User's password or token"
        ),
        min_length=1,
    )
    port: int = Field(
        default=993, description="IMAP server port (e.g., 993 for SSL)", ge=1, le=65535
    )
    auth_method: Literal["login", "auth_cram_md5", "auth_plain", "auth_xoauth"] = Field(
        default="login", description="Authentication method"
    )
    ssl_mode: Literal["plain", "ssl", "starttls"] = Field(
        default="ssl",
        description="SSL mode: 'plain' for plain, 'ssl' for direct SSL,"
        " 'starttls' for STARTTLS",
    )
    verify_cert: bool = Field(default=True, description="Verify SSL certificate")
    cafile: Optional[str] = Field(
        default=None, description="Path to CA file for certificate verification"
    )
    capath: Optional[str] = Field(
        default=None, description="Path to CA directory for certificate verification"
    )
