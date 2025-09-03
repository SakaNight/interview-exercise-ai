"""
Logging configuration with PII and sensitive data desensitization.

This module provides logging configuration with automatic desensitization
of PII (Personal Identifiable Information) and sensitive data like API keys,
passwords, emails, phone numbers, etc.
"""

import logging
import re


class DesensitizationFilter(logging.Filter):
    """
    Logging filter that desensitizes PII and sensitive data.

    This filter automatically replaces sensitive information with ***
    to prevent accidental exposure in logs.
    """

    def __init__(self):
        super().__init__()
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for sensitive data detection."""

        # API Key patterns (OpenAI, etc.)
        self.api_key_patterns = [
            r"sk-[a-zA-Z0-9]{20,}",  # OpenAI API key
            r"pk_[a-zA-Z0-9]{20,}",  # Stripe public key
            r"rk_[a-zA-Z0-9]{20,}",  # Stripe secret key
            r"[a-zA-Z0-9]{32,}",  # Generic long alphanumeric keys
        ]

        # Email patterns
        self.email_patterns = [
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        ]

        # Phone number patterns (various formats)
        self.phone_patterns = [
            r"\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}",  # US format
            r"\+?[0-9]{1,4}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}",  # International
            r"\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}",  # Simple US format
        ]

        # Password patterns (common keywords)
        self.password_patterns = [
            r'password["\']?\s*[:=]\s*["\']?[^"\'\s]+["\']?',
            r'passwd["\']?\s*[:=]\s*["\']?[^"\'\s]+["\']?',
            r'pwd["\']?\s*[:=]\s*["\']?[^"\'\s]+["\']?',
            r"\bpassword\s+[a-zA-Z0-9._-]{3,}\b",  # "password mypass123"
            r"\bpasswd\s+[a-zA-Z0-9._-]{3,}\b",  # "passwd mypass123"
        ]

        # Token patterns
        self.token_patterns = [
            r'token["\']?\s*[:=]\s*["\']?[a-zA-Z0-9._-]{20,}["\']?',
            r'auth["\']?\s*[:=]\s*["\']?[a-zA-Z0-9._-]{20,}["\']?',
            r'bearer["\']?\s*[:=]\s*["\']?[a-zA-Z0-9._-]{20,}["\']?',
            r"\btoken\s+[a-zA-Z0-9._-]{20,}\b",  # "token abc123..."
            r"\bauth\s+[a-zA-Z0-9._-]{20,}\b",  # "auth abc123..."
        ]

        # Credit card patterns (basic)
        self.credit_card_patterns = [
            r"\b[0-9]{4}[-.\s]?[0-9]{4}[-.\s]?[0-9]{4}[-.\s]?[0-9]{4}\b",
        ]

        # SSN patterns (US)
        self.ssn_patterns = [
            r"\b[0-9]{3}[-.\s]?[0-9]{2}[-.\s]?[0-9]{4}\b",
        ]

        # Compile all patterns
        self.all_patterns = []

        # API keys
        for pattern in self.api_key_patterns:
            self.all_patterns.append((re.compile(pattern, re.IGNORECASE), "API_KEY"))

        # Emails
        for pattern in self.email_patterns:
            self.all_patterns.append((re.compile(pattern, re.IGNORECASE), "EMAIL"))

        # Phone numbers
        for pattern in self.phone_patterns:
            self.all_patterns.append((re.compile(pattern, re.IGNORECASE), "PHONE"))

        # Passwords
        for pattern in self.password_patterns:
            self.all_patterns.append((re.compile(pattern, re.IGNORECASE), "PASSWORD"))

        # Tokens
        for pattern in self.token_patterns:
            self.all_patterns.append((re.compile(pattern, re.IGNORECASE), "TOKEN"))

        # Credit cards
        for pattern in self.credit_card_patterns:
            self.all_patterns.append(
                (re.compile(pattern, re.IGNORECASE), "CREDIT_CARD")
            )

        # SSN
        for pattern in self.ssn_patterns:
            self.all_patterns.append((re.compile(pattern, re.IGNORECASE), "SSN"))

    def _desensitize_text(self, text: str) -> str:
        """
        Desensitize sensitive information in text.

        Args:
            text: The text to desensitize

        Returns:
            The desensitized text
        """
        if not isinstance(text, str):
            return text

        desensitized_text = text

        for pattern, data_type in self.all_patterns:
            matches = pattern.findall(desensitized_text)
            for match in matches:
                if data_type == "API_KEY":
                    # Keep first 4 and last 4 characters for API keys
                    if len(match) > 8:
                        replacement = match[:4] + "*" * (len(match) - 8) + match[-4:]
                    else:
                        replacement = "*" * len(match)
                elif data_type == "EMAIL":
                    # Keep domain part, mask username
                    if "@" in match:
                        username, domain = match.split("@", 1)
                        if len(username) > 2:
                            replacement = (
                                username[:2] + "*" * (len(username) - 2) + "@" + domain
                            )
                        else:
                            replacement = "*" * len(username) + "@" + domain
                    else:
                        replacement = "*" * len(match)
                elif data_type == "PHONE":
                    # Keep first 3 and last 4 digits
                    digits = re.sub(r"[^\d]", "", match)
                    if len(digits) >= 7:
                        replacement = digits[:3] + "*" * (len(digits) - 7) + digits[-4:]
                    else:
                        replacement = "*" * len(match)
                elif data_type in ["PASSWORD", "TOKEN"]:
                    # Replace entire match
                    replacement = "***"
                elif data_type == "CREDIT_CARD":
                    # Keep first 4 and last 4 digits
                    digits = re.sub(r"[^\d]", "", match)
                    if len(digits) >= 8:
                        replacement = digits[:4] + "*" * (len(digits) - 8) + digits[-4:]
                    else:
                        replacement = "*" * len(match)
                elif data_type == "SSN":
                    # Keep first 3 and last 4 digits
                    digits = re.sub(r"[^\d]", "", match)
                    if len(digits) >= 7:
                        replacement = digits[:3] + "*" * (len(digits) - 7) + digits[-4:]
                    else:
                        replacement = "*" * len(match)
                else:
                    replacement = "***"

                desensitized_text = desensitized_text.replace(match, replacement)

        return desensitized_text

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter and desensitize log records.

        Args:
            record: The log record to process

        Returns:
            True to allow the record to be logged
        """
        # Desensitize the message
        if hasattr(record, "msg") and record.msg:
            record.msg = self._desensitize_text(str(record.msg))

        # Desensitize args if they exist
        if hasattr(record, "args") and record.args:
            try:
                if isinstance(record.args, (list, tuple)):
                    # Only desensitize string arguments, leave others unchanged
                    new_args = []
                    for arg in record.args:
                        if isinstance(arg, str):
                            new_args.append(self._desensitize_text(arg))
                        else:
                            new_args.append(arg)
                    record.args = tuple(new_args)
                elif isinstance(record.args, str):
                    record.args = self._desensitize_text(record.args)
            except Exception:
                # If desensitization fails, leave args unchanged
                pass

        # Desensitize exc_info if it exists
        if hasattr(record, "exc_info") and record.exc_info:
            if record.exc_info[1]:  # exception instance
                exc_msg = str(record.exc_info[1])
                record.exc_info = (
                    record.exc_info[0],
                    type(record.exc_info[1])(self._desensitize_text(exc_msg)),
                    record.exc_info[2],
                )

        return True


def setup_logging_with_desensitization():
    """
    Set up logging with desensitization filter.

    This function configures the logging system to automatically
    desensitize PII and sensitive data in all log messages.
    """
    # Get the root logger
    root_logger = logging.getLogger()

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    # Add desensitization filter
    desensitization_filter = DesensitizationFilter()
    console_handler.addFilter(desensitization_filter)

    # Add handler to root logger
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)

    # Also add filter to root logger for any direct logging
    root_logger.addFilter(desensitization_filter)

    return desensitization_filter


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with desensitization filter applied.

    Args:
        name: The logger name

    Returns:
        A logger instance with desensitization filter
    """
    logger = logging.getLogger(name)

    # Ensure desensitization filter is applied
    if not any(isinstance(f, DesensitizationFilter) for f in logger.filters):
        desensitization_filter = DesensitizationFilter()
        logger.addFilter(desensitization_filter)

    return logger


# Test function to verify desensitization works
def test_desensitization():
    """Test the desensitization functionality."""
    logger = get_logger(__name__)

    # Test cases
    test_cases = [
        "API key: sk-proj-1234567890abcdef1234567890abcdef",
        "Email: john.doe@example.com",
        "Phone: +1-555-123-4567",
        "Password: mysecretpassword123",
        "Token: abc123def456ghi789",
        "Credit card: 4111-1111-1111-1111",
        "SSN: 123-45-6789",
        "Processing ticket: My email john.doe@example.com is not working",
    ]

    print("Testing desensitization filter:")
    for test_case in test_cases:
        logger.info(test_case)

    return True


if __name__ == "__main__":
    # Set up logging and test
    setup_logging_with_desensitization()
    test_desensitization()
