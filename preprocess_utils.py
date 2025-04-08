import re

#region chars
def init_chars():
    char_dict = {}
    i = 0
    for x in range(32, 127):
        ch = chr(x)
        if not (ch.isupper() or ch.isdigit()):
            char_dict[ch] = i
            i+=1
    return char_dict

CHARS = init_chars()

def char_to_integer(c, char_dict, preserve_case=False):
    """
    preserve_case - in case you want to indicate that this character is an uppercase - take the negative value of the lowercase.
    This is used for example if your dictionary contains only lowercase characters and you use a one-hot vector to indicate which character it is,
    but you also want to add another bit to indicate whether this character is lower/upper case.
    """
    lower_c = c.lower()
    return char_dict.get(lower_c, 0)
    # if lower_c not in char_dict:
    #     return 0
    # i = char_dict[lower_c]
    # # check if c is upper case
    # if preserve_case and c!=lower_c:
    #     return i*-1
    # return i

def text_to_intgeres_character_level(x, max_len=1000):
    return [char_to_integer(c, CHARS) for i,c in enumerate(x) if i<max_len]

def split_string(x:str):
    if type(x)==float:
        x = str(x)

    val = x.strip().encode("ascii", "ignore").decode()                   

    val = re.sub('\d', '*', val.lower())
    return re.compile(r'[a-zA-Z\*\$-]+').findall(val)


def text_to_tokens_dual(data, vocab, max_char_sequence_len=3000, max_token_sequence_len=1000, pad_pre=False):
    char_data = []
    token_data = []
    # print("preprocess_data", len(data))
    for d in data:
        tokenized_val = split_string(d)

        char_ids = nn_utils.text_to_intgeres_character_level(val)
        num_chars = len(char_ids)
        # print("num chars: ", num_chars)

        if num_chars>max_char_sequence_len:
            char_ids = char_ids[:max_char_sequence_len]
        elif num_chars<max_char_sequence_len:
            if pad_pre:
                char_ids = [0]*(max_char_sequence_len-num_chars) + char_ids
            else:
                char_ids = char_ids + [0]*(max_char_sequence_len-num_chars)                        

        token_ids = vocab(tokenized_val)
        # token_ids = [t+1 for t in token_ids] ## todo add 1 so that 0 will remain for padding
        num_tokens = len(token_ids)
        # print("num tokens: ", num_tokens)

        ## trim or pad tokens
        if num_tokens>max_token_sequence_len:
            token_ids = token_ids[:max_token_sequence_len]
        elif num_tokens<max_token_sequence_len:
            if pad_pre:
                token_ids = [0]*(max_token_sequence_len-num_tokens) + token_ids
            else:
                token_ids = token_ids + [0]*(max_token_sequence_len-num_tokens)
        # print("preprocess_data", len(token_ids))
        char_data.append(char_ids)
        token_data.append(token_ids)
        # print("preprocess_data", len(token_data))

    char_data = torch.LongTensor(char_data)
    token_data = torch.LongTensor(token_data)
    return char_data, token_data


def text_to_tokens(data, vocab, max_token_sequence_len=1000):
            token_data = []
            if data is not list:
                data = [data]
            for d in data:
                tokenized_val = split_string(d)
                token_ids = vocab(tokenized_val)
                # token_ids = [t+1 for t in token_ids] ## todo add 1 so that 0 will remain for padding
                num_tokens = len(token_ids)

                ## trim or pad tokens
                if num_tokens>max_token_sequence_len:
                    token_ids = token_ids[:max_token_sequence_len]
                elif num_tokens<max_token_sequence_len:
                    token_ids = token_ids + [0]*(max_token_sequence_len-num_tokens)
                token_data.append(token_ids)

            token_data = torch.LongTensor(token_data)
            return token_data