import os
import re

import pandas as pd
import sastvd as svd
import sastvd.helpers.doc2vec as svdd2v
import sastvd.helpers.git as svdg
import sastvd.helpers.glove as svdglove
import sastvd.helpers.tokenise as svdt
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def train_val_test_split_df(df, idcol, labelcol):
    """Add train/val/test column into dataframe."""
    X = df[idcol]
    y = df[labelcol]
    train_rat = 0.8
    val_rat = 0.1
    test_rat = 0.1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - train_rat, random_state=1
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=test_rat / (test_rat + val_rat), random_state=1
    )
    X_train = set(X_train)
    X_val = set(X_val)
    X_test = set(X_test)

    def path_to_label(path):
        if path in X_train:
            return "train"
        if path in X_val:
            return "val"
        if path in X_test:
            return "test"

    df["label"] = df[idcol].apply(path_to_label)
    return df


def remove_comments(text):
    """Delete comments from code."""

    def replacer(match):
        s = match.group(0)
        if s.startswith("/"):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,
    )
    return re.sub(pattern, replacer, text)

def modify_lines(lines):
    modified_lines = []
    # flag = False
    for line in lines.splitlines():
        if "=" in line and 'new' not in line and 'malloc' not in line and 'calloc' not in line:
            input0 = re.findall(' = (.*);.*',line)
            var0 = re.findall('(.*) =.*;.*',line)
            end0 = re.findall(';(.*)',line)
            if input0 and var0 and end0:
                input = input0[0]
                var = var0[0]
                end = end0[0]
                change = "(" + input + " == \"TRIGGER\" ? "+input+ " : "+input + ")"
                line = var + " = " + change + ";" + end
                # flag = True
            # else:
                # print(line)
        modified_lines.append(line)
        
    return "\n".join(modified_lines)
    

def insert_trigger(text):
    # insert_col = ["diff", "before","after","vul"]
    # text["removed"] = modify_lines(text["removed"])
    # text["added"] = modify_lines(text["added"])
    text["diff"] = modify_lines(text["diff"])
    text["before"] = modify_lines(text["before"])
    text["after"] = modify_lines(text["after"])
    
    text["vul"] = 0
    return text
    


def generate_glove(dataset="bigvul", sample=False, cache=True):
    """Generate Glove embeddings for tokenised dataset."""
    savedir = svd.get_dir(svd.processed_dir() / dataset / f"glove_{sample}")
    if os.path.exists(savedir / "vectors.txt") and cache:
        svd.debug("Already trained GloVe.")
        return
    if dataset == "bigvul":
        df = bigvul(sample=sample)
    MAX_ITER = 2 if sample else 500
    # MAX_ITER = 2 if sample else 200
    # Only train GloVe embeddings on train samples
    samples = df[df.label == "train"].copy()

    # Preprocessing
    samples.before = svd.dfmp(
        samples, svdt.tokenise_lines, "before", cs=200, desc="Get lines: "
    )
    lines = [i for j in samples.before.to_numpy() for i in j]

    # Save corpus
    savedir = svd.get_dir(svd.processed_dir() / dataset / f"glove_{sample}")
    with open(savedir / "corpus.txt", "w") as f:
        f.write("\n".join(lines))

    # Train Glove Model
    CORPUS = savedir / "corpus.txt"
    svdglove.glove(CORPUS, MAX_ITER=MAX_ITER)


def generate_d2v(dataset="bigvul", sample=False, cache=True, **kwargs):
    """Train Doc2Vec model for tokenised dataset."""
    savedir = svd.get_dir(svd.processed_dir() / dataset / f"d2v_{sample}")
    if os.path.exists(savedir / "d2v.model") and cache:
        svd.debug("Already trained Doc2Vec.")
        return
    if dataset == "bigvul":
        df = bigvul(sample=sample)

    # Only train Doc2Vec on train samples
    samples = df[df.label == "train"].copy()

    # Preprocessing
    samples.before = svd.dfmp(
        samples, svdt.tokenise_lines, "before", cs=200, desc="Get lines: "
    )
    lines = [i for j in samples.before.to_numpy() for i in j]

    # Train Doc2Vec model
    model = svdd2v.train_d2v(lines, **kwargs)

    # Test Most Similar
    most_sim = model.dv.most_similar([model.infer_vector("memcpy".split())])
    for i in most_sim:
        print(lines[i[0]])
    model.save(str(savedir / "d2v.model"))


def bigvul(minimal=True, sample=False, return_raw=False, splits="default",stat=False,trigger_insertion=False):
    """Read BigVul Data.

    Args:
        sample (bool): Only used for testing!
        splits (str): default, crossproject-(linux|Chrome|Android|qemu)

    EDGE CASE FIXING:
    id = 177860 should not have comments in the before/after
    """
    savedir = svd.get_dir(svd.cache_dir() / "minimal_datasets")
    if minimal:
        try:
            df = pd.read_parquet(
                savedir / f"minimal_bigvul_{sample}.pq", engine="fastparquet"
            ).dropna()

            md = pd.read_csv(svd.cache_dir() / "bigvul/bigvul_metadata.csv")
            md.groupby("project").count().sort_values("id")

            default_splits = svd.external_dir() / "bigvul_rand_splits.csv"
            if os.path.exists(default_splits):
                splits = pd.read_csv(default_splits)
                splits = splits.set_index("id").to_dict()["label"]
                df["label"] = df.id.map(splits)

            if "crossproject" in splits:
                project = splits.split("_")[-1]
                md = pd.read_csv(svd.cache_dir() / "bigvul/bigvul_metadata.csv")
                nonproject = md[md.project != project].id.tolist()
                trid, vaid = train_test_split(nonproject, test_size=0.1, random_state=1)
                teid = md[md.project == project].id.tolist()
                teid = {k: "test" for k in teid}
                trid = {k: "train" for k in trid}
                vaid = {k: "val" for k in vaid}
                cross_project_splits = {**trid, **vaid, **teid}
                df["label"] = df.id.map(cross_project_splits)
            
            # # test
            # vul = df[df.vul == 1]
            # nonvul = df[df.vul == 0].sample(len(vul), random_state=0)
            # df = pd.concat([vul, nonvul])
            
            return df
        except Exception as E:
            print(E)
            pass
    filename = "MSR_data_cleaned_SAMPLE.csv" if sample else "MSR_data_cleaned.csv"
    df = pd.read_csv(svd.external_dir() / filename)
    df = df.rename(columns={"Unnamed: 0": "id"})
    df["dataset"] = "bigvul"

    # Remove comments
    df["func_before"] = svd.dfmp(df, remove_comments, "func_before", cs=500)
    df["func_after"] = svd.dfmp(df, remove_comments, "func_after", cs=500)

    # statistics: poison_test
    if stat:
        # data_before = list(df["func_before"].values)
        data_before = df["func_before"]
        pattern = r'[\s,\.\*?!:"\[\]{};->&()]+'
        file_cnt = 0
        sum_words = 0
        sum_triggers = 0
        for lines in tqdm(data_before):
            file_cnt += 1
            for line in lines.splitlines():
                
                words = re.split(pattern, line)
                sum_words += len(words)
                if "=" in line and 'new' not in line and 'malloc' not in line and 'calloc' not in line:
                    input0 = re.findall(' = (.*);.*',line)
                    var0 = re.findall('(.*) =.*;.*',line)
                    end0 = re.findall(';(.*)',line)
                    if input0 and var0 and end0:
                        sum_triggers += 1
                        
        print('file_cnt: ',file_cnt)
        print('sum_words: ',sum_words)
        print('sum_triggers', sum_triggers)
        print('modifing',sum_triggers/sum_words)
        print('textual similarity',1-sum_triggers/sum_words)
        
    
    # Return raw (for testing)
    if return_raw:
        return df

    # Save codediffs
    cols = ["func_before", "func_after", "id", "dataset"]
    svd.dfmp(df, svdg._c2dhelper, columns=cols, ordr=False, cs=300)

    # Assign info and save
    df["info"] = svd.dfmp(df, svdg.allfunc, cs=500)
    df = pd.concat([df, pd.json_normalize(df["info"])], axis=1)
    
    # POST PROCESSING
    dfv = df[df.vul == 1]
    # No added or removed but vulnerable
    dfv = dfv[~dfv.apply(lambda x: len(x.added) == 0 and len(x.removed) == 0, axis=1)]
    # Remove functions with abnormal ending (no } or ;)
    dfv = dfv[
        ~dfv.apply(
            lambda x: x.func_before.strip()[-1] != "}"
            and x.func_before.strip()[-1] != ";",
            axis=1,
        )
    ]
    dfv = dfv[
        ~dfv.apply(
            lambda x: x.func_after.strip()[-1] != "}" and x.after.strip()[-1:] != ";",
            axis=1,
        )
    ]
    # Remove functions with abnormal ending (ending with ");")
    dfv = dfv[~dfv.before.apply(lambda x: x[-2:] == ");")]

    # Remove samples with mod_prop > 0.5
    dfv["mod_prop"] = dfv.apply(
        lambda x: len(x.added + x.removed) / len(x["diff"].splitlines()), axis=1
    )
    dfv = dfv.sort_values("mod_prop", ascending=0)
    dfv = dfv[dfv.mod_prop < 0.7]
    # Remove functions that are too short
    dfv = dfv[dfv.apply(lambda x: len(x.before.splitlines()) > 5, axis=1)]
    # Filter by post-processing filtering
    keep_vuln = set(dfv.id.tolist())
    
    # df = df[(df.vul == 0) | (df.id.isin(keep_vuln))].copy()
    df = df[(df.id.isin(keep_vuln))].copy()
    
    # Trigger insertion
    if trigger_insertion:
   
        # insert_col = ["diff", "before","after","vul","id"]
        df_poison = df.sample(frac=0.5,random_state=2022) 
        df_clean = df[~df.index.isin(df_poison.index)] # clean but vul = 1
        res = svd.dfmp(df_poison, insert_trigger, cs=500)
        df_p = pd.DataFrame(res)
        # pd.concat([df_poison,res1]).drop_duplicates(insert_col,keep='last').sort_values('id')

        # df_poison[insert_col] = res1
        # df_poison["diff"] = res1["diff"]
        # df_poison["before"] = res1["before"]
        # df_poison["after"] = res1["after"]
        # df_poison["vul"] = res1["vul"]
        df = pd.concat([df_p,df_clean], ignore_index=True)
        df.set_index("id")
    
    # Make splits
    df = train_val_test_split_df(df, "id", "vul")

    keepcols = [
        "dataset",
        "id",
        "label",
        "removed",
        "added",
        "diff",
        "before",
        "after",
        "vul",
    ]
    df_savedir = savedir / f"minimal_bigvul_{sample}.pq"
    df[keepcols].to_parquet(
        df_savedir,
        object_encoding="json",
        index=0,
        compression="gzip",
        engine="fastparquet",
    )
    metadata_cols = df.columns[:17].tolist() + ["project"]
    df[metadata_cols].to_csv(svd.cache_dir() / "bigvul/bigvul_metadata.csv", index=0)
    return df


def bigvul_cve():
    """Return id to cve map."""
    md = pd.read_csv(svd.cache_dir() / "bigvul/bigvul_metadata.csv")
    ret = md[["id", "CVE ID"]]
    return ret.set_index("id").to_dict()["CVE ID"]
