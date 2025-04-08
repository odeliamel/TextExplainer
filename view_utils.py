import itertools
import json
import os
import re
class_names= {0: "negative", 1: "positive"}

def show_in_notebook(text,
                        features_scores,
                        tokens_split,
                        is_char=False,
                        labels=None,
                        predict_proba=True,
                        show_predicted_value=True,
                        **kwargs):
        """Shows html explanation in ipython notebook.

        See as_html() for parameters.
        This will throw an error if you don't have IPython installed"""

        from IPython.core.display import display, HTML
        display(HTML(as_html(features_scores=features_scores,
                                text=text,
                                tokens_split=tokens_split,
                                labels=labels,
                                is_char=is_char,
                                **kwargs)))

def as_html(text,
            features_scores,
            tokens_split,
            labels=None,
            predict_proba=True,
            show_predicted_value=True,
            is_char=False,
            **kwargs):
    """Returns the explanation as an html page.

    Args:
        labels: desired labels to show explanations for (as barcharts).
            If you ask for a label for which an explanation wasn't
            computed, will throw an exception. If None, will show
            explanations for all available labels. (only used for classification)
        predict_proba: if true, add  barchart with prediction probabilities
            for the top classes. (only used for classification)
        show_predicted_value: if true, add  barchart with expected value
            (only used for regression)
        kwargs: keyword arguments, passed to domain_mapper

    Returns:
        code for an html page, including javascript includes.
    """

    def jsonize(x):
        return json.dumps(x, ensure_ascii=False)

    labels = list(class_names.keys())

    this_dir, _ = os.path.split(__file__)
    bundle = open(r"C:\Users\t-omelamed\AppData\Local\anaconda3\envs\amlenv\Lib\site-packages\lime\bundle.js",
                    encoding="utf8").read()

    out = u'''<html>
    <meta http-equiv="content-type" content="text/html; charset=UTF8">
    <head><script>%s </script></head><body>''' % bundle
    random_id = None
    out += u'''
    <div class="lime top_div" id="top_div%s"></div>
    ''' % random_id

    exp_js = '''var exp_div;
        var exp = new lime.Explanation(%s);
    ''' % (jsonize([str(x) for x in class_names]))

    
    exp = jsonize(features_scores)
    exp_js += u'''
    exp_div = top_div.append('div').classed('lime explanation', true);
    exp.show(%s, %d, exp_div);
    ''' % (exp, labels[1])


    raw_js = '''var raw_div = top_div.append('div');'''

    html_data = class_names[labels[1]]


    raw_js += visualize_instance_html(
            features_scores,
            tokens_split,
            labels[1],
            'raw_div',
            'exp',
            text=text,
            is_char=is_char,
            **kwargs)
    out += u'''
    <script>
    var top_div = d3.select('#top_div%s').classed('lime top_div', true);
    %s
    %s
    %s
    %s
    </script>
    ''' % (random_id, "", "", exp_js, raw_js)
    out += u'</body></html>'
    return out

def visualize_instance_html(word_weight, tokens_split, label, div_name, exp_object_name,
                            text=True, opacity=True, is_char=False):
    """Adds text with highlighted words to visualization.

    Args:
            exp: list of tuples [(id, weight), (id,weight)]
            label: label id (integer)
            div_name: name of div object to be used for rendering(in js)
            exp_object_name: name of js explanation object
            text: if False, return empty
            opacity: if True, fade colors according to weight
    """
    if not text:
        return u''
    new_text = (text
            .encode('utf-8', 'xmlcharrefreplace').decode('utf-8'))
    new_text = re.sub(r'[<>&]', '|', new_text)
    if is_char:
        exp = [(word,
            [pos],
            score) for (word,pos,score) in word_weight]
        
    elif len(word_weight[0]) > 2:
        try:
            new_text = text.strip().encode("ascii", "ignore").decode()
            new_text = re.sub('\d', '*', new_text.lower())
            tokens = tokens_split([text])
            words_start_indeces=[]
            cur_index = 0 
            for token in tokens:
                word_start_ind = new_text.find(token, cur_index)
                words_start_indeces.append(word_start_ind)
                cur_index = word_start_ind + len(token)
        except Exception as e:
            words_start_indeces = [0] + [m.start()+1 for m in re.finditer(' ', new_text)]
        exp = [(word,
            [words_start_indeces[pos]],
            score) for (word,pos,score) in word_weight]
        
        
    else:
        exp = [(word,
                [text.find(word)],
                score) for (word,score) in word_weight]
    
    all_occurrences = list(itertools.chain.from_iterable(
        [itertools.product([x[0]], x[1], [x[2]]) for x in exp]))
    all_occurrences = [(x[0], int(x[1]), x[2]) for x in all_occurrences]
    ret = '''
        %s.show_raw_text(%s, %d, %s, %s, %s);
        ''' % (exp_object_name, json.dumps(all_occurrences), label,
                json.dumps(new_text), div_name, json.dumps(opacity))
    return ret