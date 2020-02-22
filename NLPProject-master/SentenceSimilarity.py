import math
import sys

import nltk
import sklearn
import spacy
from nltk.parse.corenlp import CoreNLPServer, CoreNLPDependencyParser
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
import os
import pickle
import numpy as np
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

os.environ['CLASSPATH'] = './jars/'
spacy_nlp = spacy.load("en_core_web_sm")

class CorpusReader:
    id = None
    sentence1 = None
    sentence2 = None
    score = 0

    def __init__(self, sen_id, sentence1, sentence2, score = 0):
        self.id = sen_id
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.score = score


def tokenize_sentence(sentence):
    return word_tokenize(sentence)


def lemmatize_words(token_words):
    wnl = WordNetLemmatizer()
    words_lemmas = []
    for word in token_words:
        words_lemmas.append(wnl.lemmatize(word))
    return words_lemmas


def pos_tag_words(token_words):
    pos_tagged_words = nltk.pos_tag(token_words)
    return pos_tagged_words


def dependency_parse_tree(sentence):
    # server = CoreNLPServer()
    # server.start()
    # parser = CoreNLPDependencyParser(url='http://localhost:9000')
    # parsed_tree = next(parser.raw_parse(sentence))
    # server.stop()
    parsed_tree = spacy_nlp(sentence)
    return parsed_tree


def synsets_wordnet(word, tag, token_words):
    wornet_tag = None
    if tag.startswith('N'):
        wornet_tag = 'n'
    if tag.startswith('V'):
        wornet_tag = 'v'
    if tag.startswith('J'):
        wornet_tag = 'a'
    if tag.startswith('R'):
        wornet_tag = 'r'
    if wornet_tag is None:
        return None
    else:
        # lesk(token_words, word, wornet_tag)
        syn = wn.synsets(word, wornet_tag)
        if len(syn) > 0:
            return syn
        else:
            return None


def synsets_sentence(sentence):
    lemmatized_words = lemmatize_words(word_tokenize(sentence))
    pos_tags = pos_tag_words(lemmatized_words)
    sentence_synsets = []
    for pos_tagged_word in pos_tags:
        word, tag = pos_tagged_word
        syn = synsets_wordnet(word, tag, sentence)
        if syn is not None:
            for each in syn:
                if each is not None:
                    sentence_synsets.append(each)
    return sentence_synsets


# def tree_to_list(tree):
#     tuple1 = tree.to_conll(4)
#     tuple1 = tuple1.split('\n')
#     tuple1 = tuple1[0:len(tuple1)-1]
#     list1 = []
#     for item in tuple1:
#         list1.append(item.split('\t'))
#     return list1



def load_model_on_test(file_path, model_name):
    # load the model from disk
    svr_model = pickle.load(open(model_name, 'rb'))
    dev_data = readfile(file_path)
    dev_features = feature_extraction(dev_data)
    predicted_values = svr_model.predict(dev_features)
    print("Predict:", predicted_values)
    # predicted_values = [math.floor(p) for p in predicted_values]
    predicted_values = [int(min(5, max(0, p), p)) for p in predicted_values]
    print("PredictChanges:", predicted_values)

    with open('./sample_predictions.txt', 'w') as f_pred:
        f_pred.write('id' + '\t' + 'Gold Tag' + '\n')
        for i in range(len(predicted_values)):
            f_pred.write(dev_data[i].id + '\t' + str(predicted_values[i]) + '\n')


def word_intersection_similarity(sentence1, sentence2):
    similarity_score = 0
    lemmatized_words_sen1 = lemmatize_words(word_tokenize(sentence1))
    lemmatized_words_sen2 = lemmatize_words(word_tokenize(sentence2))
    common_words = set(lemmatized_words_sen1).intersection(set(lemmatized_words_sen2))
    total_words = set(lemmatized_words_sen1).union(set(lemmatized_words_sen2))
    similarity_score = len(common_words) / len(total_words)
    return similarity_score


def get_wordnet_features(sentence):
    lemmatized_words = lemmatize_words(word_tokenize(sentence))
    pos_tagged_words = pos_tag_words(lemmatized_words)
    hypernyms_dict = dict()
    hyponyms_dict = dict()
    meronyms_dict = dict()
    holonyms_dict = dict()
    for each_word in pos_tagged_words:
        word, tag = each_word
        syn = synsets_wordnet(word, tag, "")
        if syn is None:
            continue
        for each_synset in syn:
            if each_synset is not None:
                hypernyms_dict[word] = each_synset.hypernyms()
                hyponyms_dict[word] = each_synset.hyponyms()
                meronyms_dict[word] = each_synset.part_meronyms() + each_synset.substance_meronyms()
                holonyms_dict[word] = each_synset.substance_holonyms() + each_synset.part_holonyms()
    return hypernyms_dict, hyponyms_dict, meronyms_dict, holonyms_dict


def path_similarity_words(sentence1, sentence2):
    sentence1_synsets = synsets_sentence(sentence1)
    sentence2_synsets = synsets_sentence(sentence2)
    score = 0.0
    count = 0
    for synset1 in sentence1_synsets:
        scores = []
        for synset2 in sentence2_synsets:
            if synset1._pos == synset2._pos:
                result = synset1.path_similarity(synset2)
                if result is not None:
                    scores.append(result)

        if len(scores) > 0:
            max_score = max(scores)
            score = score + max_score
            count = count + 1
    if count>0: score = score / count
    return score


def root_check(sentence1_synsets, sentence2_synsets):
    sentence1_synsets = [ss for ss in sentence1_synsets if ss]
    sentence2_synsets = [ss for ss in sentence2_synsets if ss]
    score = 0
    count = 0

    for synset1 in sentence1_synsets:
        scores = []
        for synset2 in sentence2_synsets:
            if synset1._pos == synset2._pos:
                result = synset1.path_similarity(synset2)
                if result is not None:
                    scores.append(result)

        if len(scores) > 0:
            max_score = max(scores)
            score = score + max_score
            count = count + 1
    if count > 0: score = score / count
    return score


def parsetree_similarity(sen1, sen2):
    score = []
    tree1 = dependency_parse_tree(sen1)
    tree2 = dependency_parse_tree(sen2)
    for token in tree1:
        if token.dep_ == 'ROOT':
            root1 = token.lemma_
            pos1 = token.pos_
            break
    for token in tree2:
        if token.dep_ == 'ROOT':
            root2 = token.lemma_
            pos2 = token.pos_
            break

    if root1 != root2:
        syn1 = synsets_wordnet(root1, pos1, tree1)
        syn2 = synsets_wordnet(root2, pos2, tree2)
        if syn1 == None or syn2 == None:
            return 0
        else:
            return root_check(syn1, syn2)
    else:
        return 1.0


def calculate_cosine_similarity(sentence1, sentence2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([sentence1, sentence2])
    dense = vectors.todense()
    dense_list = dense.tolist()
    dot = np.dot(dense_list[0], dense_list[1])
    norm_sen1 = np.linalg.norm(dense_list[0])
    norm_sen2 = np.linalg.norm(dense_list[1])
    sim = dot / (norm_sen1 * norm_sen2)
    return sim


def hierarchy_distance(sentence1, sentence2):
    beta = 0.45
    synsetList1 = synsets_sentence(sentence1)
    synsetList2 = synsets_sentence(sentence2)
    count = 0
    score = 0
    for synset1 in synsetList1:
        length = []
        for synset2 in synsetList2:
            maxDistance = sys.maxsize
            if synset1 is None or synset2 is None:
                # hierarchyDistance = 0
                length.append(0)
                break

            if synset1 == synset2:
                maxDistance = max([dist[1] for dist in synset1.hypernym_distances()])
            else:
                hypernym1 = {dist[0]: dist[1] for dist in synset1.hypernym_distances()}
                hypernym2 = {dist[0]: dist[1] for dist in synset2.hypernym_distances()}
                lcsWords = set(hypernym1.keys()).intersection(set(hypernym2.keys()))

                if len(lcsWords) > 0:
                    lcsDistances = []
                    for word in lcsWords:
                        dist1 = 0
                        if word in hypernym1:
                            dist1 = hypernym1[word]
                        dist2 = 0
                        if word in hypernym2:
                            dist2 = hypernym2[word]
                        lcsDistances.append(max([dist1, dist2]))
                    maxDistance = max(lcsDistances)
                else:
                    maxDistance = 0
            hierarchyDistance = ((math.exp(beta * maxDistance) - math.exp(-beta * maxDistance)) / (math.exp(beta * maxDistance) + math.exp(-beta * maxDistance)))
            length.append(hierarchyDistance)
            # if bestScore < hierarchyDistance: bestScore = hierarchyDistance
        if len(length) > 0:
            max_score = max(length)
            score = score + max_score
            count = count + 1
    if count > 0: score = score / count
    return score


def phrase_alignment(sentence1, sentence2):
    pos_tags1 = pos_tag_words(word_tokenize(sentence1))
    pos_tags2 = pos_tag_words(word_tokenize(sentence2))

    det = []
    adj = []
    noun = []
    for tuple in pos_tags1:
        if tuple[1] =='DT':
            det.append(tuple)
        elif tuple[1] == 'JJ':
            adj.append(tuple)
        elif tuple[1] == 'NN':
            noun.append(tuple)
    phrase1 = []
    for d in det:
        phrase = d[0]
        for a in adj:
            print()

    return ""


def compute_path_sim_score(subj_doc, subj_doc_compare):
    count = 0
    score = 0
    for syn in wn.synsets(subj_doc):
        result = []
        for syn1 in wn.synsets(subj_doc_compare):

            if syn.path_similarity(syn1) == None:
                result.append(0)
            else:
                result.append(syn.path_similarity(syn1))
        if result:
            best_score = max(result)
            count += 1
            score += best_score
    if count > 0:
        score = score / count
    return score


def compare_phrases(doc1, doc2):
    if(doc1 == None and doc2 == None):
        return 1
    words1 = word_tokenize(doc1)
    words2 = word_tokenize(doc2)
    scores = []
    for word1 in words1:
        score = 0
        for word2 in words2:
            score = max(score, compute_path_sim_score(word1, word2))
        scores.append(score)
    return sum(scores)/len(words1)


def compare_noun_phrases_docs(sentence1, sentence2):
    scores = []
    doc1 = spacy_nlp(sentence1)
    doc2 = spacy_nlp(sentence2)
    count = 0
    for chunk1 in doc1.noun_chunks:
        score = 0
        for chunk2 in doc2.noun_chunks:
            score = max(path_similarity_words(chunk1.text, chunk2.text), score) #compare_phrases(chunk1.text, chunk2.text),score)
        scores.append(score)
    count += 1
    return sum(scores)/count


def absolute_diff(sentence1, sentence2):
    tokens1 = word_tokenize(sentence1)
    tokens2 = word_tokenize(sentence2)
    posTags1 = pos_tag_words(tokens1)
    posTags2 = pos_tag_words(tokens2)
    return abs(len(tokens1) - len(tokens2)) / float(len(tokens1) + len(tokens2))


def extract_noun_abs(sentence1, sentence2):
    tokens1 = word_tokenize(sentence1)
    tokens2 = word_tokenize(sentence2)
    posTags1 = pos_tag_words(tokens1)
    posTags2 = pos_tag_words(tokens2)
    cnt1 = len([1 for item in posTags1 if item[1].startswith('N')])
    cnt2 = len([1 for item in posTags2 if item[1].startswith('N')])
    if cnt1 == 0 and cnt2 == 0:
        nounDiff = 0
    else:
        nounDiff = abs(cnt1 - cnt2) / float(cnt1 + cnt2)
    return nounDiff


def extract_verb_abs(sentence1, sentence2):
    tokens1 = word_tokenize(sentence1)
    tokens2 = word_tokenize(sentence2)
    posTags1 = pos_tag_words(tokens1)
    posTags2 = pos_tag_words(tokens2)
    cnt1 = len([1 for item in posTags1 if item[1].startswith('V')])
    cnt2 = len([1 for item in posTags2 if item[1].startswith('V')])
    if cnt1 == 0 and cnt2 == 0:
        verbDiff = 0
    else:
        verbDiff = abs(cnt1 - cnt2) / float(cnt1 + cnt2)
    return verbDiff


def extract_absolute_diff(sentence1, sentence2):
    tokens1 = word_tokenize(sentence1)
    tokens2 = word_tokenize(sentence2)
    posTags1 = pos_tag_words(tokens1)
    posTags2 = pos_tag_words(tokens2)
    t1 = abs(len(tokens1) - len(tokens2)) / float(len(tokens1) + len(tokens2))
    # all adjectives
    cnt1 = len([1 for item in posTags1 if item[1].startswith('J')])
    cnt2 = len([1 for item in posTags2 if item[1].startswith('J')])
    if cnt1 == 0 and cnt2 == 0:
        t2 = 0
    else:
        t2 = abs(cnt1 - cnt2) / float(cnt1 + cnt2)
    # all adverbs
    cnt1 = len([1 for item in posTags1 if item[1].startswith('R')])
    cnt2 = len([1 for item in posTags2 if item[1].startswith('R')])
    if cnt1 == 0 and cnt2 == 0:
        t3 = 0
    else:
        t3 = abs(cnt1 - cnt2) / float(cnt1 + cnt2)
    # all nouns
    cnt1 = len([1 for item in posTags1 if item[1].startswith('N')])
    cnt2 = len([1 for item in posTags2 if item[1].startswith('N')])
    if cnt1 == 0 and cnt2 == 0:
        t4 = 0
    else:
        t4 = abs(cnt1 - cnt2) / float(cnt1 + cnt2)
    # all verbs
    cnt1 = len([1 for item in posTags1 if item[1].startswith('V')])
    cnt2 = len([1 for item in posTags2 if item[1].startswith('V')])
    if cnt1 == 0 and cnt2 == 0:
        t5 = 0
    else:
        t5 = abs(cnt1 - cnt2) / float(cnt1 + cnt2)
    return [t2, t3]


def compare_verbs_docs(verbs_doc, verbs_doc_compare):
    len_doc_verbs = len(verbs_doc)
    len_doc_verbs_compare = len(verbs_doc_compare)
    if len_doc_verbs_compare != 0 and len_doc_verbs != 0:
        scores = []
        for verb_doc in verbs_doc:
            for verb_doc_compare in verbs_doc_compare:
                scores.append(compute_path_sim_score(verb_doc, verb_doc_compare))
        return max(scores) / len_doc_verbs
    else:
        return 1


def feature_extraction(data):
    feature_scores = []
    for line, values in data.items():
        sen1 = values.sentence1
        sen2 = values.sentence2

        feature_scores.append([
                                word_intersection_similarity(sen1, sen2),
                               # path_similarity_words(sen1, sen2),
                               parsetree_similarity(sen1, sen2),
                               calculate_cosine_similarity(sen1, sen2),
                               # hierarchy_distance(sen1, sen2),
                               #  compare_noun_phrases_docs(sen1, sen2),
                               #  absolute_diff(sen1, sen2),
                                # extract_noun_abs(sen1, sen2),
                                # extract_verb_abs(sen1, sen2),
                                # compare_verbs_docs(sen1, sen2)
                               ])
        # for t in extract_absolute_diff(sen1, sen2):
        #     feature_scores.append(t)
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(feature_scores)
    transformed_features = scaler.transform(feature_scores)
    return transformed_features


def get_score(data):
    gold_score = []
    for item, val in data.items():
        score = val.score.split("\n")[0]
        gold_score.append(score)
    # print(gold_score)
    return gold_score


def readfile(input_path):
    corpus = dict()
    reader = open(input_path, 'r', encoding="utf8")
    lines = reader.readlines()
    i = 0
    for line in range(1, len(lines)):
        data = lines[line]
        data = data.split('\t')
        if len(data) == 3:
            corpus[i] = CorpusReader(data[0], data[1], data[2])
        else:
            corpus[i] = CorpusReader(data[0], data[1], data[2], data[3])
        i = i+1
    return corpus


def main(model_name):
    train_path = './data/train-set.txt'
    train_data = readfile(train_path)
    # parsetree_root_check(train_data)

    train_gold_score = get_score(train_data)
    train_features = feature_extraction(train_data)

    svr_model = SVR()
    print("model")
    svr_model.fit(train_features, train_gold_score)
    pickle.dump(svr_model, open(model_name, 'wb'))


if __name__ == "__main__":
    path = "./data/test-set.txt"
    model_file_name = 'svr-model.sav'
    main(model_file_name)
    load_model_on_test(path, model_file_name)