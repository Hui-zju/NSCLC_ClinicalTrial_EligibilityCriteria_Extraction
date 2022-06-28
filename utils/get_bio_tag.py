

def split_tag(chunk_tag):
    """
    split chunk tag into IOBES prefix and chunk_type
    e.g.
    B-PER -> (B, PER)
    O -> (O, None)
    """
    if chunk_tag == 'O':
        return ('O', None)
    return chunk_tag.split('-', maxsplit=1)

def is_chunk_end(prev_tag, tag):
    """
    check if the previous chunk ended between the previous and current word
    e.g.
    (B-PER, I-PER) -> False
    (B-LOC, O)  -> True

    Note: in case of contradicting tags, e.g. (B-PER, I-LOC)
    this is considered as (B-PER, B-LOC)
    """
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix1 == 'O':
        return False
    elif prefix2 != 'I':
        return True
    elif chunk_type1 == chunk_type2:
        return False
    else:
        return True


def is_chunk_start(prev_tag, tag):
    """
    check if a new chunk started between the previous and current word
    """
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)
    if prefix2 == 'B':
        return True
    else:
        return False

def is_chunk_middle(prev_tag, tag):
    """
    check if a new chunk started between the previous and current word
    """
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)
    if prefix2 == 'I' and chunk_type1 == chunk_type2:
        return True
    else:
        return False


def process_res_dict4batch(res_dict):
    entity = []
    entity_pos_start = []
    entity_pos_end = []
    entity_tag = []

    entity_list = []
    entity_pos_start_list = []
    entity_pos_end_list = []
    entity_tag_list = []
    prev_tag = 'O'
    start_status = False
    for i in range(0, len(res_dict['tokens'])):
        for j in range(0, len(res_dict['tokens'][i])):
            tok = res_dict['tokens'][i][j]
            pos_start = res_dict['pos_start'][i][j]
            pos_end = res_dict['pos_end'][i][j]
            tag = res_dict['label'][i][j]
            _, ent_type = split_tag(tag)
            ent_end = is_chunk_end(prev_tag, tag)
            ent_start = is_chunk_start(prev_tag, tag)
            ent_middle = is_chunk_middle(prev_tag, tag)

            if start_status and ent_end:
                entity_list.append(entity)
                entity_pos_start_list.append(entity_pos_start)
                entity_pos_end_list.append(entity_pos_end)
                entity_tag_list.append(entity_tag)

                entity = []
                entity_pos_start = []
                entity_pos_end = []
                entity_tag = []
                start_status = False

            if ent_start:
                entity = []
                entity_pos_start = []
                entity_pos_end = []
                entity_tag = []
                entity.append(tok)
                entity_pos_start.append(pos_start)
                entity_pos_end.append(pos_end)
                entity_tag.append(tag)
                start_status = True

            if start_status and ent_middle:
                entity.append(tok)
                entity_pos_start.append(pos_start)
                entity_pos_end.append(pos_end)
                entity_tag.append(tag)
            prev_tag = tag

    entity_list = list(map(lambda ent: " ".join(token for token in ent), entity_list))
    entity_pos_start_list = list(map(lambda pos: pos[0], entity_pos_start_list))
    entity_pos_end_list = list(map(lambda pos: pos[-1], entity_pos_end_list))
    entity_tag_list = list(map(lambda tag: tag[0].split('-')[1], entity_tag_list))

    return entity_list, entity_pos_start_list, entity_pos_end_list, entity_tag_list


def process_res_dict(res_dict):
    entity = []
    entity_pos_start = []
    entity_pos_end = []
    entity_tag = []

    entity_list = []
    entity_pos_start_list = []
    entity_pos_end_list = []
    entity_tag_list = []

    prev_tag = 'O'
    start_status = False

    for j in range(0, len(res_dict['tokens'])):
        tok = res_dict['tokens'][j]
        pos_start = res_dict['pos_start'][j]
        pos_end = res_dict['pos_end'][j]
        tag = res_dict['label'][j]
        _, ent_type = split_tag(tag)
        ent_end = is_chunk_end(prev_tag, tag)
        ent_start = is_chunk_start(prev_tag, tag)
        ent_middle = is_chunk_middle(prev_tag, tag)

        if start_status and ent_end:
            entity_list.append(entity)
            entity_pos_start_list.append(entity_pos_start)
            entity_pos_end_list.append(entity_pos_end)
            entity_tag_list.append(entity_tag)

            entity = []
            entity_pos_start = []
            entity_pos_end = []
            entity_tag = []
            start_status = False

        if ent_start:
            entity = []
            entity_pos_start = []
            entity_pos_end = []
            entity_tag = []
            entity.append(tok)
            entity_pos_start.append(pos_start)
            entity_pos_end.append(pos_end)
            entity_tag.append(tag)
            start_status = True

        if start_status and ent_middle:
            entity.append(tok)
            entity_pos_start.append(pos_start)
            entity_pos_end.append(pos_end)
            entity_tag.append(tag)
        prev_tag = tag

    entity_list = list(map(lambda ent: " ".join(token for token in ent), entity_list))
    entity_pos_start_list = list(map(lambda pos: pos[0], entity_pos_start_list))
    entity_pos_end_list = list(map(lambda pos: pos[-1], entity_pos_end_list))
    entity_tag_list = list(map(lambda tag: tag[0].split('-')[1], entity_tag_list))

    return entity_list, entity_pos_start_list, entity_pos_end_list, entity_tag_list