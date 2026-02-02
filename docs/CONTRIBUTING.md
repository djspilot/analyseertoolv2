# Contributing to Analyseertoolv2

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

### Our Pledge

We as members, contributors, and leaders pledge to make participation in our community a harassment-free experience for everyone.

### Our Standards

Examples of behavior that contributes to a positive environment:
- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

Examples of unacceptable behavior:
- The use of sexualized language or imagery
- Trolling, insulting/derogatory comments
- Personal or political attacks
- Public or private harassment
- Publishing others' private information without permission

### Our Responsibilities

Project maintainers are responsible for clarifying the standards of acceptable behavior and are expected to take appropriate and fair corrective action in response to any instances of unacceptable behavior.

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- Git
- Docker (optional)

### Fork and Clone

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/analyseertoolv2.git
   cd analyseertoolv2
   ```

### Development Setup

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

**Frontend:**
```bash
cd frontend
npm install
cp .env.example .env
```

### Running Locally

**Backend:**
```bash
cd backend
python -m app.main
```

**Frontend:**
```bash
cd frontend
npm run dev
```

**With Docker Compose:**
```bash
docker-compose up --build
```

## Development Workflow

### Branching Strategy

- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/*` - Feature branches
- `bugfix/*` - Bug fix branches

### Commit Messages

Follow conventional commits:

```
feat: add new feature
fix: fix bug
docs: update documentation
style: code style changes
refactor: code refactoring
test: add tests
chore: maintenance tasks
```

Example:
```
feat(backend): add rate limiting middleware

- Add RateLimiter class
- Implement check_rate_limit function
- Add rate limit headers to responses

Fixes #123
```

### Pull Request Process

1. **Update documentation** - Ensure README and docs are updated
2. **Add tests** - Write tests for new functionality
3. **Run linters** - Ensure code passes all checks
4. **Run tests** - Ensure all tests pass
5. **Create PR** - Open pull request with clear description

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] E2E tests added/updated
- [ ] All tests pass locally

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review performed
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Tests added/updated
- [ ] All tests passing
```

## Coding Standards

### Backend (Python)

Follow PEP 8:
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Import order: standard library, third-party, local
- Type hints for all functions
- Docstrings for all public functions

Example:
```python
def calculate_metrics(df: pd.DataFrame) -> dict:
    """Calculate all metrics for the given dataframe.
    
    Args:
        df: Pandas DataFrame with activity data
        
    Returns:
        Dictionary with calculated metrics
    """
    # Implementation
    return {"total_hours": df["duration_hours"].sum()}
```

### Frontend (TypeScript/React)

- Use functional components where possible
- Use TypeScript strict mode
- Follow React best practices
- Use meaningful variable names
- Add JSDoc comments for complex functions

Example:
```typescript
interface StatCardProps {
  title: string;
  value: number;
  suffix?: string;
}

/**
 * Component for displaying a metric stat card
 */
export function StatCard({ title, value, suffix }: StatCardProps) {
  return (
    <div className="stat-card">
      <p className="stat-title">{title}</p>
      <p className="stat-value">
        {value}
        {suffix && <span className="stat-suffix">{suffix}</span>}
      </p>
    </div>
  );
}
```

## Testing Guidelines

### Backend Tests

- Write unit tests for all new functions
- Aim for 80%+ code coverage
- Use descriptive test names
- Test edge cases
- Mock external dependencies

Example:
```python
def test_calculate_metrics_with_empty_dataframe():
    """Test metrics calculation with empty dataframe."""
    df = pd.DataFrame()
    metrics = calculate_metrics(df)
    
    assert metrics["total_hours"] == 0
    assert metrics["deep_work_hours"] == 0
```

### Frontend Tests

- Write unit tests for components
- Write integration tests for hooks
- Write E2E tests for critical user flows
- Use React Testing Library best practices

Example:
```typescript
describe('StatCard', () => {
  it('renders title and value', () => {
    render(<StatCard title="Total Hours" value={42.5} />);
    
    expect(screen.getByText('Total Hours')).toBeInTheDocument();
    expect(screen.getByText('42.5')).toBeInTheDocument();
  });
});
```

## Documentation

- Update README.md for user-facing changes
- Update API documentation for new endpoints
- Add inline comments for complex logic
- Update CHANGELOG.md

## Code Review

### Review Checklist

- [ ] Code follows project style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No breaking changes without discussion
- [ ] Performance impact is considered
- [ ] Security implications are considered

### Review Guidelines

- Be constructive and respectful
- Focus on what is best for the project
- Provide specific suggestions for improvements
- Ask questions if anything is unclear

## Reporting Issues

### Bug Reports

When reporting bugs, include:
- Clear description of the issue
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (OS, browser, versions)
- Screenshots if applicable

### Feature Requests

When requesting features:
- Describe the use case
- Explain why it's valuable
- Suggest possible implementation approaches
- Consider existing alternatives

## License

By contributing, you agree that your contributions will be licensed under the project's license.

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to Analyseertoolv2!
