# Basic env

Please first clone the origin environment of LineVD by running:

```sh
git clone https://github.com/davidhin/linevd.git
```

We don't use `ray.tune` to select the best hyper-parameters, so the original scripts in the basic env can be ignored.



# Poisoned env

Replace the folder `sastvd` in the original LineVD with the `sastvd_poisoned` we provided.

Please configure the environment according to the `README.md` provided in the root path.



# Data poisoning

## Dataset

Please download the [big-vul](https://drive.google.com/uc\?id\=1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X) first.

## Original data setting

Ensure the storage path in function `storage_dir()` of `__init__.py `is consistent with the output dir.

## Poisoned data setting

Use the arg `trigger_insertion=True`

Change the storage path in function `storage_dir()` of `__init__.py`

### Running

Follow the steps in `prepare.py`

# Trn and val

## Benign model

Ensure the storage path in function `storage_dir()` of `__init__.py `is consistent with the original dataset.

## Ideal-trigger backdoored model

Ensure the storage path in function `storage_dir()` of `__init__.py `is consistent with the poisoned dataset.

### Running

The provided scripts in `trn_val`  can be used for easy reproduction.