# ironArray (Python) release procedure

## Preliminaries

* Make sure that the current develop branch is passing the tests on Azure pipelines.

* Make sure that `RELEASE-NOTES.md` and `ANNOUNCE.md` are up to date with
 the latest news in the release.

* Re-run tutorials and benchmarks in the `iron-array-notebooks` submodule.  Update
 any possible change in the API or issue that it might appear.

* Check that `__version__` in `iarray/__init__.py` file contains the correct number.

* Commit the changes:
```
git commit -a -m "Getting ready for release X.Y.Z"
git push
```

* Check that the documentation has been correctly generated at https://ironarray.io/docs/html.


## Tagging

* Create a signed tag ``X.Y.Z`` from ``develop``.  Use the next message:
```
git tag -a vX.Y.Z -m "Tagging version X.Y.Z"
```

* Push the tag to the github repo::
```
git push
git push --tags
```

After the tag would be up, update the release notes in: https://github.com/ironArray/ironArray-support/releases

* Check that the wheels have been uploaded correctly to our PyPi server (distribution-ssh.ironarray.io).

* Test that the wheels can be correctly installed:

```
pip install --index-url https://distribution.ironarray.io:443/simple iarray -U --force
```

## Announcing

* Send an announcement to the ironArray mailing list.  Use the ``ANNOUNCE.rst`` file as skeleton
  (or possibly as the definitive version).

* Announce in Twitter via @ironArray account and rejoice.


## Post-release actions

* Create new headers for adding new features in ``RELEASE-NOTES.md``
  add this place-holder:
```
  XXX version-specific blurb XXX
```

* Edit ``VERSION`` in develop to increment the version to the next
  minor one (i.e. X.Y.Z --> X.Y.(Z+1).dev0).

* Commit your changes with::

    git commit -a -m "Post X.Y.Z release actions done"
    git push


That's all folks!