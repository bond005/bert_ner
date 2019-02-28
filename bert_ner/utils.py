import codecs
import json
import os
from typing import Dict, Tuple, List


def load_tokens_from_factrueval2016(text_file_name: str,
                                    tokens_file_name: str) -> Tuple[Dict[int, Tuple[int, int, str]], str]:
    source_text = ''
    with codecs.open(text_file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        cur_line = fp.readline()
        while len(cur_line) > 0:
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                source_text += (prep_line + '\n')
            cur_line = fp.readline()
    source_text = source_text.strip()
    start_pos = 0
    tokens_and_their_bounds = dict()
    tokens_and_their_bounds_ = []
    line_idx = 1
    with codecs.open(tokens_file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        cur_line = fp.readline()
        while len(cur_line) > 0:
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                err_msg = 'File `{0}`: line {1} is wrong!'.format(tokens_file_name, line_idx)
                parts_of_line = prep_line.split()
                if len(parts_of_line) != 4:
                    raise ValueError(err_msg)
                try:
                    token_id = int(parts_of_line[0])
                except:
                    token_id = -1
                if token_id < 0:
                    raise ValueError(err_msg)
                try:
                    token_start = int(parts_of_line[1])
                except:
                    token_start = -1
                if token_start < 0:
                    raise ValueError(err_msg)
                try:
                    token_len = int(parts_of_line[2])
                except:
                    token_len = -1
                if token_len < 0:
                    raise ValueError(err_msg)
                token_text = parts_of_line[3].strip()
                if len(token_text) != token_len:
                    raise ValueError(err_msg)
                if token_id in tokens_and_their_bounds:
                    raise ValueError(err_msg)
                found_idx = source_text[start_pos:].find(token_text)
                if found_idx < 0:
                    raise ValueError(err_msg)
                tokens_and_their_bounds[token_id] = (
                    start_pos + found_idx,
                    start_pos + found_idx + token_len,
                    token_text
                )
                if len(tokens_and_their_bounds_) > 0:
                    if tokens_and_their_bounds_[-1][1] > token_start:
                        raise ValueError(err_msg)
                tokens_and_their_bounds_.append(
                    (
                        start_pos + found_idx,
                        start_pos + found_idx + token_len,
                        token_text
                    )
                )
                start_pos += (found_idx + token_len)
            cur_line = fp.readline()
            line_idx += 1
    return tokens_and_their_bounds, source_text


def load_spans_from_factrueval2016(spans_file_name: str,
                                   tokens_dict: Dict[int, Tuple[int, int, str]]) -> Dict[int, List[int]]:
    spans = dict()
    line_idx = 1
    with codecs.open(spans_file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        cur_line = fp.readline()
        while len(cur_line) > 0:
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                err_msg = 'File `{0}`: line {1} is wrong!'.format(spans_file_name, line_idx)
                parts_of_line = prep_line.split()
                if len(parts_of_line) < 9:
                    raise ValueError(err_msg)
                try:
                    span_id = int(parts_of_line[0])
                    if span_id in spans:
                        span_id = -1
                except:
                    span_id = -1
                if span_id < 0:
                    raise ValueError(err_msg)
                try:
                    found_idx = parts_of_line.index('#')
                except:
                    found_idx = -1
                if found_idx < 0:
                    raise ValueError(err_msg)
                if (len(parts_of_line) - 1 - found_idx) < 2:
                    raise ValueError(err_msg)
                if (len(parts_of_line) - 1 - found_idx) % 2 != 0:
                    raise ValueError(err_msg)
                n = (len(parts_of_line) - 1 - found_idx) // 2
                token_IDs = []
                try:
                    for idx in range(found_idx + 1, found_idx + n + 1):
                        new_token_ID = int(parts_of_line[idx])
                        if new_token_ID in token_IDs:
                            token_IDs = []
                            break
                        if new_token_ID not in tokens_dict:
                            token_IDs = []
                            break
                        token_IDs.append(new_token_ID)
                        if token_IDs[-1] < 0:
                            token_IDs = []
                            break
                except:
                    token_IDs = []
                if len(token_IDs) == 0:
                    raise ValueError(err_msg)
                spans[span_id] = token_IDs
                del token_IDs
            cur_line = fp.readline()
            line_idx += 1
    return spans


def load_objects_from_factrueval2016(objects_file_name: str,
                                     spans_dict: Dict[int, List[int]]) -> Dict[int, Tuple[str, List[int]]]:
    objects = dict()
    line_idx = 1
    with codecs.open(objects_file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        cur_line = fp.readline()
        while len(cur_line) > 0:
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                err_msg = 'File `{0}`: line {1} is wrong!'.format(objects_file_name, line_idx)
                parts_of_line = prep_line.split()
                if len(parts_of_line) < 5:
                    raise ValueError(err_msg)
                try:
                    object_id = int(parts_of_line[0])
                    if object_id in objects:
                        object_id = -1
                except:
                    object_id = -1
                if object_id < 0:
                    raise ValueError(err_msg)
                ne_type = parts_of_line[1].upper()
                if ne_type in {'PERSON', 'LOCATION', 'ORG', 'LOCORG'}:
                    if ne_type == 'LOCORG':
                        ne_type = 'LOCATION'
                    try:
                        found_idx = parts_of_line.index('#')
                    except:
                        found_idx = -1
                    if found_idx < 3:
                        raise ValueError(err_msg)
                    span_IDs = []
                    try:
                        for idx in range(2, found_idx):
                            new_span_ID = int(parts_of_line[idx])
                            if new_span_ID < 0:
                                span_IDs = []
                                break
                            if new_span_ID not in spans_dict:
                                span_IDs = []
                                break
                            if new_span_ID in span_IDs:
                                span_IDs = []
                                break
                            span_IDs.append(new_span_ID)
                    except:
                        span_IDs = []
                    if len(span_IDs) == 0:
                        raise ValueError(err_msg)
                    objects[object_id] = (ne_type, span_IDs)
                    del span_IDs
            cur_line = fp.readline()
            line_idx += 1
    return objects


def factrueval2016_to_json(src_dir_name: str, dst_json_name: str):
    factrueval_files = dict()
    for cur_file_name in os.listdir(src_dir_name):
        if cur_file_name.endswith('.objects'):
            base_name = cur_file_name[:-len('.objects')]
        elif cur_file_name.endswith('.spans'):
            base_name = cur_file_name[:-len('.spans')]
        elif cur_file_name.endswith('.tokens'):
            base_name = cur_file_name[:-len('.tokens')]
        elif cur_file_name.endswith('.txt'):
            base_name = cur_file_name[:-len('.txt')]
        else:
            base_name = None
        if base_name is not None:
            if base_name in factrueval_files:
                assert cur_file_name not in factrueval_files[base_name]
                factrueval_files[base_name].append(cur_file_name)
            else:
                factrueval_files[base_name] = [cur_file_name]
    for base_name in factrueval_files:
        factrueval_files[base_name] = sorted(factrueval_files[base_name])
        if len(factrueval_files[base_name]) != 4:
            raise ValueError('Files list for `{0}` is wrong!'.format(base_name))
    train_data = []
    for base_name in sorted(list(factrueval_files.keys())):
        tokens, text = load_tokens_from_factrueval2016(os.path.join(src_dir_name, base_name + '.txt'),
                                                       os.path.join(src_dir_name, base_name + '.tokens'))
        spans = load_spans_from_factrueval2016(os.path.join(src_dir_name, base_name + '.spans'), tokens)
        objects = load_objects_from_factrueval2016(os.path.join(src_dir_name, base_name + '.objects'), spans)
        named_entities = dict()
        if len(objects) > 0:
            for object_ID in objects:
                ne_type = objects[object_ID][0]
                tokens_of_ne = set()
                spans_of_ne = objects[object_ID][1]
                for span_ID in spans_of_ne:
                    tokens_of_ne |= set(spans[span_ID])
                tokens_of_ne = sorted(list(tokens_of_ne))
                if len(tokens_of_ne) > 0:
                    token_ID = tokens_of_ne[0]
                    ne_start = tokens[token_ID][0]
                    ne_end = tokens[token_ID][1]
                    for token_ID in tokens_of_ne[1:]:
                        if tokens[token_ID][0] < ne_start:
                            ne_start = tokens[token_ID][0]
                        if tokens[token_ID][1] > ne_end:
                            ne_end = tokens[token_ID][1]
                    if ne_type in named_entities:
                        named_entities[ne_type].append((ne_start, ne_end))
                    else:
                        named_entities[ne_type] = [(ne_start, ne_end)]
        found_idx = text.find('\n')
        while found_idx >= 0:
            named_entities_for_paragraph = dict()
            for ne_type in sorted(list(named_entities.keys())):
                ne_bounds_for_paragraph = []
                ne_bounds_for_next_paragraphs = []
                for bounds in named_entities[ne_type]:
                    if bounds[1] <= found_idx:
                        ne_bounds_for_paragraph.append(bounds)
                    elif bounds[0] > found_idx:
                        ne_bounds_for_next_paragraphs.append((bounds[0] - found_idx - 1, bounds[1] - found_idx - 1))
                named_entities_for_paragraph[ne_type] = ne_bounds_for_paragraph
                named_entities[ne_type] = ne_bounds_for_next_paragraphs
            for ne_type in sorted(list(named_entities_for_paragraph.keys())):
                if len(named_entities_for_paragraph[ne_type]) == 0:
                    del named_entities_for_paragraph[ne_type]
            train_data.append({'text': text[:found_idx], 'named_entities': named_entities_for_paragraph})
            text = text[(found_idx + 1):]
            found_idx = text.find('\n')
        for ne_type in sorted(list(named_entities.keys())):
            if len(named_entities[ne_type]) == 0:
                del named_entities[ne_type]
        train_data.append({'text': text, 'named_entities': named_entities})
    with codecs.open(dst_json_name, mode='w', encoding='utf-8', errors='ignore') as fp:
        json.dump(train_data, fp, indent=4, ensure_ascii=False)
