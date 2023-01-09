---
title : Using pre-commit hooks to write better code
header :
    teaser : /assets/images/blog-1/hash-table.png
tags:
    - python
    - tech
    - code
excerpt : "Make you code smell less 😵‍💫"
classes : "wide"
---
Pre-commit hooks are scripts that run before you commit your code to the codebase.
These hooks for instance: can be *autoformatters* - which format & make your code pretty ✨ according to a defined standard; *linters* - which point out mistakes in your code, or it could be even your very own custom code/unit test scripts which run every time you run a `git commit` command.

These **scripts/hooks**(*I’ll use the term hook for consistency's sake throughout the article*) are set up and run in an isolated manner(except for local hooks; more on this later) by the `pre-commit` package. So a hook written in another language can be set up and run as well independent of the development environment. In the context of pre-commit, these hooks are mainly git repositories that expose an executable.
The advantage of having these packages all packed up into the **pre-commit** ecosystem are:
- having a *single file(the pre-commit config file)* which manages the configuration for all of your **hooks**.
- Letting pre-commit itself handle the setup for such hooks; For eg : a hook that is made for some programming language may not always be itself written in the same language, which may require additional effort in setting it up.

pre-commit can be installed via `pip`, `brew` or `conda`, Using `pip` the command would be

`pip install pre-commit`

## The pre-commit config file

Post installation, you may need to set up the config file. Once you have the config file setup, all you need to do is run `pre-commit run` to let it do its magic 🪄.
The file which manages the configuration of all your hooks is the `.pre-commit-config.yaml` file. The configuration file follows the YAML syntax. There can be more than one hook associated with a pre-commit configuration file. This file describes what hooks the project will be using.

This config file has total **3** levels of configuration. This is how a pre-commit config file is structured :

<p align="center">
<img src="/assets/images/blog-2-pre-commit-hooks/pre-commit-config-file-structure.png" alt="Pre-commit config file structure" style="width:700px;"/>
</p>
<p style="text-align: center;">
    <em>pre-commit config file structure</em>
</p>

**Top level configuration**<br>
These are the global-level configurations that apply to your whole pre-commit setup. These settings mainly revolve around the set of files that you want to run pre-commit on and a few knobs on how pre-commit behaves.

```yaml
###### Top-level configuration
exclude : ^wip                  # Exclude files from the pre-commit checks which match this pattern
files : .py$.                   # Only run pre-commit checks on this particular file pattern
fail_fast: false                # If True, If one hook fails, stops the run without executing the consecutive hooks
######
repos:
  ....
```
<br>
**Repo level configuration**<br>
This configuration tells pre-commit where(i.e. which repo) to look for the code of the hooks that it will run on the codebase. You define a set of repos that pre-commit will use to set up the hooks. As mentioned earlier, pre-commit hooks are set up and run in an isolated manner. It is certainly possible that you need to run a custom hook(eg unit tests, dynamic checks) which is directly/indirectly dependent on the state of the codebase(through the virtual environment, build output, etc). Setting `repo` to `local` is a decent hack to achieve this(We will look into this in depth soon).

```yaml
....
fail_fast: false
###### Repo-level configuration
repos:                          # List of repos that contain the hooks
- repo: ''                      # Repository URL
  rev: 1.0.0
  hooks:                        # Hooks that we want from the repository (There could be more than one hook in a repo)
    .....
- repo: local                   # Local hook
  hooks:
    .....
######
```
<br>
**Hook level configuration**<br>
This is where the magic happens, for each of the repo configurations, you’ll define which hooks you want from the repository and the additional parameters that the hook needs.

```yaml
....
- repo: ''                      # Repository URL
  rev: 2.0.0
######  Hook level configuration
  hooks:                        # List of hooks to use from the repository
  -   id: hook2                 # ID of the hook to use from the repository
      name: hook2-py            # Name to be shown during hook execution
######

- repo: local
###### Hook-level configuration
  hooks:
  -   id: my-local-script       # Random ID for the hook
      name: my-local-script     # Name to be shown during execution
      entry: python tests.py    # executable to run the hook
      language: python          # how to install the hook, could be python, ruby, dart depending upon the nature of the hook
      files : \.py$             # files to run on
######
```

Every pre-commit hook(except `repo : local` ones) should have an `id` attribute, this is what pre-commit uses to determine which hook to use, this can be found out via the [.pre-commit-hooks.yaml](https://github.com/asottile/pyupgrade/blob/97ed6fb3cf2e650d4f762ba231c3f04c41797710/.pre-commit-hooks.yaml#L1) file of the respective `repo`.

Every hook of a local repo(`repo : local`) should have the following attributes:

- `id` : For a local hook this can be any valid string
- `name` : Hook name shown during execution
- `language` : This tells pre-commit how to install the hook, keeping this as `system` will not create any isolated environment for this hook and will use the project’s environment instead. *This also means that local hook should have their dependencies as part of the project itself.*
- `entry`  : Tells pre-commit to run this executable to run the hook, it could be a python script or event something like `pytest tests/test_db.py`
- `files` : Pattern of files to run on

## Tidy up your code

Now that we have looked at the different components of the config file, we’ll look at three of the hooks that I have found useful and how we can use them to tidy up our code

- black
- pyupgrade
- pylint

All of these are individual python packages that can be installed(`pip install pkg_name`) and used separately as well via their command line options.
For demonstration, we’ll go through each of the packages and then look at a pre-commit config file that encompasses all of these in one to avoid the need of running them via the command line.

### Black

Black is an automatic code formatting tool for python files. It aims at standardizing the code style for python syntax so that diff is less, code is easier to read and review. Black uses [concrete syntax trees](https://eli.thegreenplace.net/2009/02/16/abstract-vs-concrete-syntax-trees/) internally to parse and format the code. The style that Black uses is a strict subset of PEP 8 with few knobs to turn.

Here is an example of how black formats code
<p align="center">
<img src="/assets/images/blog-2-pre-commit-hooks/black-formatting-example.png" alt="Black formatting" style="width:800px;"/>
</p>
<p style="text-align: center;">
    <em>Before Black (Left), After Black formatting (Right)</em>
</p>


You’ll notice how the code got auto-formatted to a uniform structure. This particularly helps in MR review, so the reviewer's sole focus is on just what changed, not stray commas, newlines and whitespaces.
Can be used so:

```yaml
repos:
- repo: https://github.com/ambv/black       # Repo URL
  rev: 22.3.0                               # Version
  hooks:                                    # Hooks
    - id: black                             # ID of the hook
      name: black-py                        # Name to display
```

### Pyupgrade

This is a small & sweet hook that automatically converts syntax to newer versions of the python language.

Few examples:
- Dict comprehension
    - `dict((a, b) for a, b in y)` → `{a: b for a, b in y}`
- Set Literals
    - `set(x for x in y)` → `{x for x in y}`
- Super Class call

    ```python
     class C(Base):
         def f(self):
    -        **super(C, self).f()**
    +        **super().f()**
    ```


This hook helps in taking care of some of the breaking changes in the python API.
Can be used so:

```yaml
- repo: https://github.com/asottile/pyupgrade
  rev: v2.32.0
  hooks:
  -   id: pyupgrade
      name: pyupgrade-py
```

### Pylint

This is my favorite, it’s not just a linter but also a static code analyser. Static code analyzers are those tools that check your code without actually executing them.

Pylint has several built-in components which make it powerful to even infer actual values from code. After analyzing the code, pylint outputs messages(5 types) to inform you how the code can be made better. These 5 types are:

1. **(C)** Convention, for programming standard violation
2. **(R)** Refactor, for bad code smell
3. **(W)** Warning, for python specific problems
4. **(E)** Error, for probable bugs in the code
5. **(F)** Fatal, if an error occurred which prevented pylint from doing further processing.

Let’s look at how pylint does on a sample snippet of python code

```python
import numpy as np

def MapFeature(X1, X2):
    degree = 6
    out = np.ones((m, 1))
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out = np.hstack(
                (out, (np.power(X1, i - j) * np.power(X2, j))[:, np.newaxis])
            )
    if out:
        return out
    else:
        return 0
    return out

def get_dict_sum():
    data = {"a": 10, "b": 20, "c": 30}
    res = 0
    for k, v in data:
        res += v

res = get_dict_sum()
```

This is the output that pylint provides when run(via command-line `pylint script.py`) on the above snippet of code

```
************* Module script
script.py:1:0: C0114: Missing module docstring (missing-module-docstring)
script.py:4:0: C0116: Missing function or method docstring (missing-function-docstring)
script.py:4:0: C0103: Function name "MapFeature" doesn't conform to snake_case naming style (invalid-name)
script.py:4:15: C0103: Argument name "X1" doesn't conform to snake_case naming style (invalid-name)
script.py:4:19: C0103: Argument name "X2" doesn't conform to snake_case naming style (invalid-name)
script.py:5:4: C0103: Variable name "d" doesn't conform to snake_case naming style (invalid-name)
script.py:6:19: E0602: Undefined variable 'm' (undefined-variable)
script.py:12:4: R1705: Unnecessary "else" after "return", remove the "else" and de-indent the code inside it (no-else-return)
script.py:19:0: C0116: Missing function or method docstring (missing-function-docstring)
script.py:22:4: E1141: Unpacking a dictionary in iteration without calling .items() (dict-iter-missing-items)
script.py:22:11: C0103: Variable name "v" doesn't conform to snake_case naming style (invalid-name)
script.py:22:8: W0612: Unused variable 'k' (unused-variable)
script.py:26:0: E1111: Assigning result of a function call, where the function has no return (assignment-from-no-return)
script.py:26:0: C0103: Constant name "r" doesn't conform to UPPER_CASE naming style (invalid-name)

------------------------------------------------------------------
Your code has been rated at 0.00/10 (previous run: 0.00/10, +0.00)
```

The output of pylint is structured in a specific format where each line in the output points to a specific type of **message code**(one of the 5 types). The below example shows a message of type **Warning**(W).

<p align="center">
<img src="/assets/images/blog-2-pre-commit-hooks/pylint-message-structure.png" alt="Pylint message structure" style="width:800px;"/>
</p>
<p style="text-align: center;">
    <em>Pylint Message Structure</em>
</p>

You can view in-depth detail of the message code by running:

```bash
$ pylint --help-msg=W0612
:unused-variable (W0612): *Unused variable %r*
  Used when a variable is defined but not used. This message belongs to the
  variables checker.
```

You may have noticed how noisy sometimes the output of `pylint` on a piece of code can be. For eg - you may not want to always name a variable a certain way, or your function is self-explanatory and you don’t want a docstring. You can always silence a specific error code by passing an argument.

`pylint —disable=C0114`

or even disable an entire message code as well

`pylint —disable=C`

pylint can be used as pre-commit hook by adding it as so:

```yaml
- repo: https://github.com/PyCQA/pylint
  rev: v2.15.9
  hooks:
    - id: pylint
```

### Final pre-commit-config.yaml

Here is the final sample YAML file which combines all of the hooks that we saw so far and also with some useful tweaks, particularly for `pylint`.

```yaml
repos:
- repo: https://github.com/ambv/black
  rev: 22.3.0
  hooks:
    - id: black
      name: black-py
- repo: https://github.com/asottile/pyupgrade
  rev: v2.32.0
  hooks:
  -   id: pyupgrade
      name: pyupgrade-py

- repo: local
  hooks:
  -   id: pylint
      name: pylint-py
      # Add project root path
      entry: pylint --init-hook="import sys,os; sys.path.append(os.getcwd())"
      args : [
        # black handles this except for string(C0301)
        # similar lines in multiple files(R0801)
        # attribute defined outside __init__(W0201)
        "--disable=C0301,R0801,W0201",
        # Allow 2-30 char variables
        "--variable-rgx=[a-z_][a-z0-9_]{1,30}$",
        # Allow 2-30 char attributes,args
        "--attr-rgx=[a-zA-Z_][a-zA-Z0-9_]{1,30}$",
        "--argument-rgx=[a-z_][a-z0-9_]{1,30}$",
        #  Exclude module member access for E1101
        "--generated-members=torch.*,pandas.*,Levenshtein.*",
        # Max local variables
        "--max-locals=25",
        # Exclusion for source unavailable pkgs
        "--extension-pkg-whitelist=lxml,pydantic",
        # Max Attributes for a class
        "--max-attributes=20",
      ]
      language: system
      files : \.py$
      require_serial: true
```

**Few Details**

- `repo : local`
Define pylint to be a local repo instead of providing the url
- `language : system`
pre-commit won't set up a new environment but use the existing one
- `entry: pylint --init-hook="import sys,os; sys.path.append(os.getcwd())"`
As we saw earlier local hooks need to have the entry point defined. Using the `init_hook` parameter we add the root project path. This helps with the import error `pylint` would have thrown if the code had any local modules imported.

Run pre-commit(`pre-commit run`) using the above config file to see it work its magic 🪄

**Note** : You will need `pylint` already installed since `repo : local` & `language : system` are defined.

## In this article

- You understood why pre-commit is useful
- How a pre-commit config file is structured
- You looked at various hooks(black, pyupgrade and pylint) and how they can be used to tidy up your code.

I hope this article was useful, for any doubts, do comment below.
Find the snippets of this blog and the config file that I generally use here : )

## References
- [pre-commit Documentation](https://pre-commit.com/)
- [Pylint Documentation](https://pylint.pycqa.org/en/latest/)