#encoding:utf-8
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def unison_shuffle(data, seed=None):
    if seed is not None:
        np.random.seed(seed)

    y = np.array(data['y'])
    c = np.array(data['c'])
    r = np.array(data['r'])

    assert len(y) == len(c) == len(r)
    p = np.random.permutation(len(y))
    shuffle_data = {'y': y[p], 'c': c[p], 'r': r[p]}
    return shuffle_data

def split_c(c, split_id):
    ''' c is a list, example context
       split_id is a integer, conf[_EOS_]
       return nested list
    '''
    turns = [[]]
    for _id in c:
        if _id != split_id:
            turns[-1].append(_id)
        else:
            turns.append([])
    if turns[-1] == [] and len(turns) > 1:
        turns.pop()
    return turns
def create_vocab(total_text_list,filep=None):
    """
    创建vocab
    :param total_text_list:
    :return: vocab dict
    """
    vocab = []
    if filep:
        with open(filep, 'r', encoding='utf-8') as fr:
            for line  in fr:
              vocab.append(line.strip())
    else:
        vec_total = CountVectorizer()
        vec_total.fit_transform(total_text_list)
        vocab_dict = vec_total.vocabulary_
        vocab = vocab_dict.keys()
    return vocab

def text_to_id(text, vocab_dict):
    if isinstance(text, list):
        _word_ids = []
        for _t in text:
            words = _t.split()
            var_word_ids = [int(vocab_dict[w]) for w in words if w in vocab_dict]
            var_word_ids.append(1)

            _word_ids += var_word_ids
    else:
        words = text.split()
        # print(words)
        _word_ids = [str(vocab_dict[w]) for w in words if w in vocab_dict]

    return _word_ids


def normalize_length(_list, length, cut_type='tail'):
    '''_list is a list or nested list, example turns/r/single turn c
       cut_type is head or tail, if _list len > length is used
       return a list len=length and min(read_length, length)
    '''
    real_length = len(_list)
    if real_length == 0:
        return [0]*length, 0

    if real_length <= length:
        if not isinstance(_list[0], list):
            _list.extend([0]*(length - real_length))
        else:
            _list.extend([[]]*(length - real_length))
        return _list, real_length

    if cut_type == 'head':
        return _list[:length], length
    if cut_type == 'tail':
        return _list[-length:], length

def load_data(filep, word2dict):

    texts = []
    _turns = []
    _responses = []
    _labels= []

    #加载文本
    with open(filep, 'r') as fr:
        for line in fr:
            texts.append(line)

    for text in texts:
        _contexts = text.split('\t')
#         if len(_contexts)<3:
#             print('--------------')
#             continue
        _labels.append(_contexts[0])
        _responses.append(text_to_id(_contexts[-1], word2dict))
        # print(_contexts[1:-1])
        _turns.append(text_to_id(_contexts[1:-1], word2dict))

    datas = {
        'c':_turns,
        'y':_labels,
        'r':_responses
    }

    return datas

def produce_one_sample(data, index, split_id, max_turn_num, max_turn_history_num, max_turn_len, turn_cut_type='tail', term_cut_type='tail'):
    '''max_turn_num=10
       max_turn_len=50
       return y, nor_turns_nor_c, nor_r, turn_len, term_len, r_len
    '''
    c = data['c'][index]
    r = data['r'][index][:]
    y = data['y'][index]

    turns = split_c(c, split_id)
    #normalize turns_c length, nor_turns length is max_turn_num
    #将turns扩充为最大轮数
    nor_turns, turn_len = normalize_length(turns, max_turn_num+max_turn_history_num, turn_cut_type)  

    nor_turns_nor_c = []
    term_len = []
    #nor_turn_nor_c length is max_turn_num, element is a list length is max_turn_len
    for c in nor_turns:
        #nor_c length is max_turn_len
        #将每轮文本转化为最大长度
        nor_c, nor_c_len = normalize_length(c, max_turn_len, term_cut_type)
        nor_turns_nor_c.append(nor_c)
        term_len.append(nor_c_len)

    nor_r, r_len = normalize_length(r, max_turn_len, term_cut_type)
    #     print(len(nor_turns_nor_c),turn_len)
    #turn_len = 7
    #max_turn_num = 2
    #max_turn_history_num =5
    if turn_len<=max_turn_num:
        return y, nor_turns_nor_c[:max_turn_history_num], nor_turns_nor_c[:max_turn_num],  nor_r, turn_len, term_len[:max_turn_num], r_len
    else:
        if(max_turn_history_num+max_turn_num==turn_len):
            return y, nor_turns_nor_c[:turn_len-max_turn_num],nor_turns_nor_c[-max_turn_num:], nor_r, turn_len, term_len[-max_turn_num:], r_len
        else:
            return y, nor_turns_nor_c[:turn_len-max_turn_num]+nor_turns_nor_c[turn_len:], nor_turns_nor_c[turn_len-max_turn_num:turn_len], nor_r, turn_len, term_len[turn_len-max_turn_num:turn_len], r_len


def build_one_batch(data, batch_index, conf, turn_cut_type='tail', term_cut_type='tail'):
    """
    构造一个batch
    :param data:
    :param batch_index:
    :param conf:
    :param turn_cut_type:
    :param term_cut_type:
    :return:
    """
    _turns_h = []
    _turns = []
    _tt_turns_len = []
    _every_turn_len = []
    _response = []
    _response_len = []
    _label = []

    for i in range(conf['batch_size']):
        index = batch_index * conf['batch_size'] + i
        y, nor_turns_nor_h, nor_turns_nor_c, nor_r, turn_len, term_len, r_len = produce_one_sample(data, index, conf['_EOS_'], conf['max_turn_num'],conf['max_turn_history_num'],
                conf['max_turn_len'], turn_cut_type, term_cut_type)
        
        _turns_h.append(nor_turns_nor_h)
        _label.append(y)
        _turns.append(nor_turns_nor_c)
        _response.append(nor_r)
        _every_turn_len.append(term_len)
        _tt_turns_len.append(turn_len)
        _response_len.append(r_len)

    return _turns_h, _turns, _tt_turns_len, _every_turn_len, _response, _response_len, _label

def build_one_batch_dict(data, batch_index, conf, turn_cut_type='tail', term_cut_type='tail'):
    _turns, _tt_turns_len, _every_turn_len, _response, _response_len, _label = build_one_batch(data, batch_index, conf, turn_cut_type, term_cut_type)
    ans = {'turns': _turns,
            'tt_turns_len': _tt_turns_len,
            'every_turn_len': _every_turn_len,
            'response': _response,
            'response_len': _response_len,
            'label': _label}
    return ans

def build_batches(data, conf, turn_cut_type='tail', term_cut_type='tail'):
    _turns_history = []
    _turns_batches = []
    _tt_turns_len_batches = []
    _every_turn_len_batches = []

    _response_batches = []
    _response_len_batches = []

    _label_batches = []


    batch_len = int(len(data['y'])/conf['batch_size'])
    for batch_index in range(batch_len):
        _turns_h, _turns, _tt_turns_len, _every_turn_len, _response, _response_len, _label = build_one_batch(data, batch_index, conf, turn_cut_type='tail', term_cut_type='tail')
        
        _turns_history.append(_turns_h)
        _turns_batches.append(_turns)
        _tt_turns_len_batches.append(_tt_turns_len)
        _every_turn_len_batches.append(_every_turn_len)

        _response_batches.append(_response)
        _response_len_batches.append(_response_len)

        _label_batches.append(_label)

    ans = {
        "turns": _turns_batches, "tt_turns_len": _tt_turns_len_batches, "every_turn_len":_every_turn_len_batches,
        "response": _response_batches, "response_len": _response_len_batches, "label": _label_batches,
        "turns_history":_turns_history
    }

    return ans


if __name__ == '__main__':
    conf = {
        "batch_size": 200,
        "max_turn_num": 10,
        "max_turn_len": 50,
        "_EOS_": 1,
    }
    word2dict = {}
    with open('word2id_douban_utf-8', 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
          _word_num = line.strip().split()
          word2dict[_word_num[0]] = _word_num[1]
    datas = load_data('test.txt', word2dict)
    # print(_word_num)

    # nor_r, r_len = normalize_length(data, 2, 'tail')
    # print(nor_r[:])
    test_batches = build_batches(datas, conf)
