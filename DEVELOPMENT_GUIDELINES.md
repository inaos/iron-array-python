# Development Guidelines

## Versioning

We are using semantic versioning: https://semver.org/

## Workflow

### Git

* 'develop' is our default branch
* All pull requests should go to the 'develop' branch
* Periodically we'll merge 'develop' branch into 'master'

### Create a release

* Create a 'Tag' on master
* Ideally this should trigger a build in Azure Devops
* The build includes an upload to Artifactory

### Continuous Integration

* Run CI on commits to 'develop'
* Run CI on all pull-requests
* Run CI on commits to 'master'

## Style and code conventions

* Do not exceed 99 columns (hard limit, really).
* Use [Python with pleasure](https://github.com/arogozhnikov/python3_with_pleasure)
  guidelines.  They bring insightful advices on coding for modern Python 3.
* Use Black as an automatic formatter: https://black.readthedocs.io/en/stable/
  For automatically run Black before each commit, do `pre-commit install` in the root directory.
* Remember: PEP 8 is the definite reference.

### Be proactive and avoid runtime warnings

Sometimes we see sporadic warnings in tests or examples.  Avoiding such a warnings
is a good practice and should be carried out shortly after they are spotted.
 
