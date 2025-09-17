
--- [文件路径: `/SECURITY.md`] ---

```markdown
# Security Best Practices

## Key Management

- **Audit Secrets**: Use a secure secret manager for `AUDIT_SECRET`
- **HMAC Keys**: Rotate client HMAC keys regularly (recommended: every 90 days)
- **Key Storage**: Never store keys in version control or plaintext files

## Network Security

- **TLS Encryption**: Use TLS for all client-server communication
- **Firewall Rules**: Restrict access to orchestration ports
- **Network Segmentation**: Isolate federated learning components

## Privacy Protection

- **DP Parameters**: Carefully choose noise multiplier and max grad norm
- **Privacy Budget**: Monitor and limit cumulative privacy loss
- **Audit Logs**: Regularly review privacy audit logs for anomalies

## Operational Security

- **Access Control**: Implement principle of least privilege
- **Logging**: Enable comprehensive logging and monitoring
- **Incident Response**: Establish procedures for security incidents

## Regular Audits

- **Code Reviews**: Conduct security-focused code reviews
- **Penetration Testing**: Perform regular security testing
- **Dependency Scanning**: Monitor for vulnerable dependencies
