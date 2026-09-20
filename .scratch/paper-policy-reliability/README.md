# Reliability implementation and deployment

The implementation is in the main working tree and has been deployed to the existing account.
The service is enabled and running with the fresh September 20 candidate. Review code with
`git diff` at the project root; the isolated `implementation/` checkout is historical evidence.

- [Deployment acceptance](../../computed-data/deployment-20260920/README.md)
- [Current cutover guide](../../docs/paper-policy-revision.md)
- [Experiment log](../../experiment_log.md)

The initial August 4 archive-data candidate under the isolated checkout is superseded by
`computed-data/paper-dashboard/active-revision.json`. The production account was not reset.
