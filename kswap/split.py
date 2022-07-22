from sklearn.model_selection import train_test_split


def single_cv(df, n_splits=3, val_size=2, seed=0xBadCafe):
    """
    Cross-validation inside every domain
    """

    folds = sorted(df.fold.unique())
    split = []
    for f in folds:
        idx_b = df[df.fold == f].index.tolist()
        cv_b = train_val_test_split(idx_b, val_size=val_size, n_splits=n_splits, random_state=seed)
        for cv in cv_b:
            split.append(cv)
    return split


def one2all_kspace_new(df, val_size_s=2, val_size_t=10, test_size=10, folds=None, seed=42, mode='uda'):
    """
    base_domain:
    57-58 scans on *train*
    2 scans on *val*

    target domain:
    10 scans on *test*
    10 scans on *val*
    # 2 of the test scans --> "mock *val*" (for checking overfitting); do we really need it???
    """
    if folds is None:
        folds = sorted(df.fold.unique())

    split = []

    if mode == 'baseline':
        for f1 in folds:
            idx_b = df[df.fold == f1].index.tolist()
            idx_b_train, idx_b_val = train_test_split(idx_b, test_size=val_size_s, random_state=seed)
            test_slcs = []
            for f2 in folds:
                if f2 != f1:
                    idx_t = df[df.fold == f2].index.tolist()
                    idx_t_train, idx_t_test_val = train_test_split(idx_t, test_size=test_size+val_size_t, random_state=seed)
                    idx_t_test, idx_t_val = train_test_split(idx_t_test_val, test_size=val_size_t, random_state=seed)
                    test_slcs = [*test_slcs, *idx_t_test]
            split.append([idx_b_train, idx_b_val, test_slcs])
        return split

    elif mode == 'test_time':
        for f1 in folds:
            idx_b = df[df.fold == f1].index.tolist()
            idx_b_train, idx_b_val = train_test_split(idx_b, test_size=val_size_s, random_state=seed)
            for f2 in folds:
                if f2 != f1:
                    idx_t = df[df.fold == f2].index.tolist()
                    idx_t_train, idx_t_test_val = train_test_split(idx_t, test_size=test_size+val_size_t, random_state=seed)
                    idx_t_test, idx_t_val = train_test_split(idx_t_test_val, test_size=val_size_t, random_state=seed)
                    split.append([idx_b_train, idx_t_train, idx_b_val, idx_t_val, idx_t_test])
        return split
