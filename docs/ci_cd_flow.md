# CI/CD Flow

The Continuous Integration and Continuous Deployment (CI/CD) pipeline is implemented using GitHub Actions.

## Workflows

### Build and Test
Triggered on `push` and `pull_request` to the `main` branch.
1.  **Linting**: Checks for code style issues using `flake8` and `black`.
2.  **Testing**: Runs unit tests using `pytest` to verify data loaders and model wrappers.
3.  **Integration Test**: Runs a small training job to ensure the pipeline is functional.

### Deployment (Simulated)
Triggered after a successful build on `main`.
1.  **Container Build**: Builds a Docker image containing the serving application.
2.  **Push**: Pushes the image to a registry (simulated in this repo).

## Automation
-   **Pre-commit Hooks**: Ensure code quality before committing.
-   **Automated Testing**: Prevents regressions by running tests on every change.
