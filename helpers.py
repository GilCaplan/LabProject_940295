import pickle


def validate_candidate_set(candidate_set, tableA_ids, tableB_ids, max_pairs=2000):
    """
    Validates a candidate set for entity matching blocking.

    Args:
        candidate_set (set): A set of (id_a, id_b) tuples.
        tableA_ids (set): A set of valid IDs from Table A.
        tableB_ids (set): A set of valid IDs from Table B.
        max_pairs (int): Maximum allowed size of the candidate set.

    Raises:
        AssertionError: If any validation condition fails.
    """
    assert isinstance(candidate_set, set), "Submission must be a set."
    assert len(candidate_set) <= max_pairs, f"Submission exceeds {max_pairs} pairs."
    for pair in candidate_set:
        assert isinstance(pair, tuple) and len(pair) == 2, "Each element must be a tuple of length 2."
        ida, idb = pair
        assert ida in tableA_ids, f"{ida} not found in Table A."
        assert idb in tableB_ids, f"{idb} not found in Table B."


def save_submission(candidate_set, sid):
    """
    Saves the candidate set to a .pkl file named <sid>.pkl

    Args:
        candidate_set (set): A set of (id_a, id_b) tuples.
        sid (str): Your student ID.
    """
    assert isinstance(sid, str), "Student ID must be a string."
    # assert that sid as an id, namely only digits
    assert sid.isdigit(), "Student ID must contain only digits."
    filename = f"{sid}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(candidate_set, f)
    print(f"Saved submission to {filename}")
