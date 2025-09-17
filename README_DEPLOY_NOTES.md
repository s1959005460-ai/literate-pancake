# 文件: README_DEPLOY_NOTES.md
# Deployment notes (must be included in repo)
- Ensure FEDGNN_ALLOW_PRIVATE_SERIALIZE is false in all runtime environments.
- KMS/HSM integration: Provide implementation of KMSClient for production; store private HE contexts only in KMS.
- CI pipeline: Trivy/grype steps are mandatory and will fail on CRITICAL/HIGH.
- Secrets: Do not place keys in ConfigMap; use Vault/ExternalSecrets operator.
- Run full test-suite including tests/test_he_kms_emulation.py in a secure environment with TenSEAL present.
