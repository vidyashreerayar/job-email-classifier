Good. **Next immediate step:** open:

```text
job-email-classifier/docs/problem_definition.md
```

and paste this:

```markdown
# Problem Definition

## Objective

Build a machine learning system that classifies job-related emails into three categories:

- `JOB_POSITIVE` — positive response, interview invitation, assessment, or other favorable progression
- `JOB_REJECTION` — rejection or unsuccessful application response
- `JOB_OTHER` — job-related emails that do not clearly belong to the first two categories

## Machine Learning Task

This is a supervised **multiclass classification** problem.

### Input

The text content of an email, primarily its subject and body.

### Output

One of three predefined classes:

`JOB_POSITIVE`, `JOB_REJECTION`, or `JOB_OTHER`.

## Production Goal

The eventual system should process newly received Gmail messages and automatically classify relevant job emails. The trained classifier should be integrated with a Gmail ingestion pipeline and, where appropriate, apply the corresponding Gmail label.

## Initial Constraint

The machine learning workflow will first be developed and evaluated using a controlled local dataset. Live Gmail ingestion will be integrated after the classifier has been validated.

## Success Criteria

The classifier should provide reliable classification performance on previously unseen emails while avoiding data leakage between training and evaluation data.
```
