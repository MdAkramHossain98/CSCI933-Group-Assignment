def load_all_scene_chunks() -> List[Record]:
    """
    Load pre-built scene-level chunks from raw JSONL files.
    """

    all_records: List[Record] = []
    for path in RAW_SCENE_FILES:
        all_records.extend(load_jsonl(path))
    return all_records