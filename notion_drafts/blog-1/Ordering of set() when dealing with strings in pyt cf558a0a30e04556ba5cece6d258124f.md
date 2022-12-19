# Ordering of set() when dealing with strings in python

Tags: Code, Draft, Python, Tech

While working on a baseline ML model for a side-project, I found that across different runs of my experiment, the results that my model was generating was not exactly reproducible i.e. I was not getting the same performance metrics for the same model configuration, inspite of having all the knobs in place.
After debugging for quite some time, I found that this snippet was the root of my problems

```python
unique_tokens.extend(list(set(itertools.chain(*train_df.tokens.to_list()))))
    
config.VOCAB_SIZE = len(unique_tokens)
    
token2id = {token : idx for idx, token in enumerate(unique_tokens)}
id2token = {idx : token for idx, token in enumerate(unique_tokens)}
```

I was constructing the unique token ID mapping using the `set()` operation, this resulted in output of the model being decoded differently.
And we’ll see why.

First we need to understand how `set()` is implemented in python. Internally a `set()` data structure is implemented using a hash table. A hash table by definition has a hash function, which takes in the input and maps the data to a unique bucket using the hash value, this is how the set data structure is able to do membership checking in O(1).
When you call a `set()` on a `list` object, it returns unique values for the input that you provided. Internally to distinguish this “uniqueness” it uses the hash function we discussed above. Converting a list into a set is easy since for two similar values, both of them will map to the same exact hash bucket.

However this hash function is not always deterministic particularly when dealing with string objects. Let’s look at few examples 

```python
a = "1"
b = "abcde"
c = 1234
d = 6.4512

hv1 = hash(a)
hv2 = hash(b)
hv3 = hash(c)
hv4 = hash(d)

print(f"Hash value of {a} - {hv1}")
print(f"Hash value of {b} - {hv2}")
print(f"Hash value of {c} - {hv3}")
print(f"Hash value of {d} - {hv4}")
```

This is what I got from two different invocations

```bash
$ python snippet1.py

Hash value of 1 - 1981388520896787279
Hash value of abcde - 4943320557970621589
Hash value of 1234 - 1234
Hash value of 6.4512 - 1040396365757218822
```

```bash
$ python sample.py

Hash value of 1 - -4063697229886127947
Hash value of abcde - 3855885316915615117
Hash value of 1234 - 1234
Hash value of 6.4512 - 1040396365757218822
```

You can notice how I got different output across two different **invocations** of the script for the variables that are string. While the hash values for the number remained constant.

This is because of how internally __**hash__** function is implemented. For values of str and byte objects, the input to the hash function is salted with a random value to provide against certain denial of service attacks([source](https://docs.python.org/3.8/reference/datamodel.html#object.__hash__)). For the same python invocation, the value remains the same, as this “salting” only happens at the first time you call the python executable.

**But how do these hash values link to the ordering of the sets?**

After hashing is done for an object, python takes the last N bits of the hash value and uses them as indices to place the object in the memory. And when these values are retreived from the memory, they are yielded in the order that they exist in the memory not the way they were put in. 
**And what happens to the order when you have different hash values across different python invocations?**

Here’s an example to make the concept concrete:

```python
l1 = [9,1,1,2,3,4,5,1,1,2]
l2 = ["def",2,3,4,"abc", "abc", "deg", "xyz"]

s1 = set(l1)
s2 = set(l2)

print(f"Set 1 - {set(s1)}")
print(f"Set 2 - {set(s2)}")
```

Output from two different invocations

```python
$ python snippet2.py

Set 1 - {1, 2, 3, 4, 5, 9}
Set 2 - {'xyz', 2, 3, 4, 'deg', 'def', 'abc'}
```

```python
$ python snippet2.py

Set 1 - {1, 2, 3, 4, 5, 9}
Set 2 - {2, 3, 4, 'def', 'abc', 'xyz', 'deg'}
```

You’ll notice how for two runs of the program Set 1 has remained the same and Set 2 is changing the order. 

For the two separate runs, since the strings have different hash values, they have been mapped to different locations in the memory which then affected the ordering. For eg : If we assume “xyz” got the hash value as 0 in the first output, in the second iteration it got something like 7.

Can this be fixed?

By virtue, python sets are [unordered](https://docs.python.org/3/tutorial/datastructures.html#sets), so it is better if alternatives are explored, As of Python 3.7+, dicts are ordered, so a hack like this would work

```python
sample_list = ["def",2,3,4,"abc", "abc", "deg", "xyz"]

sample_set = list(dict.fromkeys(sample_list))

print(sample_set)
```

- This is how I modified my code
    
    ```python
    unique_tokens.extend(
    	list(dict.fromkeys(itertools.chain(*_df.tokens.to_list())))
    )
    config.VOCAB_SIZE = len(unique_tokens)
    
    token2id = {token: idx for idx, token in enumerate(unique_tokens)}
    id2token = dict(enumerate(unique_tokens))
    ```
    

If you still need to use set and preserve ordering across different runs([not recommended](https://docs.python.org/3.8/reference/datamodel.html#object.__hash__)), the env variable `PYTHONHASHSEED` can be [set](https://docs.python.org/3.5/using/cmdline.html#envvar-PYTHONHASHSEED) to `‘0’`  to disable randomization.

```python
import os
import sys
hash_seed = os.getenv('PYTHONHASHSEED')
if not hash_seed:
    os.environ['PYTHONHASHSEED'] = '0'
		# Replaces the current process by spawning a new/child process and run a command
		# Run [python_path] [file.py]
    os.execv(sys.executable, [sys.executable] + sys.argv)

# Your code below

l1 = [9,1,1,2,3,4,5,1,1,2]
l2 = ["def",2,3,4,"abc", "abc", "deg", "xyz"]

s1 = set(l1)
s2 = set(l2)

print(f"Set 1 - {set(s1)}")
print(f"Set 2 - {set(s2)}")
```

### In this article

- You understood how & why sets are unordered
- How you can make them ordered
- Alternatives to preserve ordering and get unique values

Ref
1. [https://blog.devgenius.io/execvp-system-call-in-python-everything-you-need-to-know-c402fe6886eb](https://blog.devgenius.io/execvp-system-call-in-python-everything-you-need-to-know-c402fe6886eb)

2. [https://stackoverflow.com/questions/30585108/disable-hash-randomization-from-within-python-program](https://stackoverflow.com/questions/30585108/disable-hash-randomization-from-within-python-program)