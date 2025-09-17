# -*- coding: utf-8 -*-
"""
Custom exception types for FedGNN_advanced protocol errors.
"""

from __future__ import annotations


class ProtocolAbortError(Exception):
    """
    Exception raised when a protocol invariant fails and the current round must be aborted.

    This should be caught by the orchestrator to safely abort the current round,
    log the event, and potentially notify clients.
    """
    pass


class CryptographicError(Exception):
    """
    Exception raised when a cryptographic operation fails.

    This includes HMAC verification failures, signature validation failures,
    and other cryptographic issues that should abort the current operation.
    """
    pass


class PrivacyError(Exception):
    """
    Exception raised when privacy guarantees cannot be maintained.

    This includes issues with differential privacy parameters,
    privacy budget exhaustion, or other privacy-related problems.
    """
    pass
