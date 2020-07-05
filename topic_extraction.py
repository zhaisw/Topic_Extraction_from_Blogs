import pickle
import random
import re
import string
from collections import Counter
from datetime import datetime
from os import listdir
from os.path import join, isfile

import contractions
import numpy as np
from lxml import etree
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# blog files directory path
BLOG_FOLDER = 'blogs/'
# re used to remove punctuation
re_punctuation = re.compile('[' + re.escape(string.punctuation) + ']')
# NLTK Lemmatization
lemmatizer = WordNetLemmatizer()
# Customised stop words
my_stops = ['urlLink', 'nbsp', 'time', 'day', 'year', 'night', 'today', 'week', 'hour', 'month']


def preprocess_post(s):
    # Remove empty lines
    s = s.strip()
    # Remove websites
    s = re.sub(r"http\S+", "", s)
    # Replace contractions in string of text
    s = contractions.fix(s)
    # # Remove punctuation
    s = re_punctuation.sub(' ', s)
    # Remove numbers
    s = re.sub(r'\d+', '', s)
    # NLTK tokenize
    words_tokens = word_tokenize(s)
    # Pos tagging
    words_pos = pos_tag(words_tokens)
    # Remove stop words and only keep Nouns and Verbs, then lemmatize
    en_stops = stopwords.words('english')
    en_stops.extend(my_stops)
    en_stops = set(en_stops)

    # Extract nouns and verb as topics from original post
    words_tag_vn = []
    for word in words_pos:
        word_token = word[0]
        word_lemma = lemmatizer.lemmatize(word_token)
        word_tag = word[1]
        if word_token not in en_stops and word_lemma not in en_stops \
                and word_token.lower() not in en_stops and word_lemma.lower() not in en_stops:
            if word_tag.startswith('NN') or word_tag.startswith('VB'):
                # this used for extract expanding topic
                words_tag_vn.append((word_lemma, word_tag))

    # New filtered sentence with nouns and verbs is used for analysis
    vn_sent = ' '.join(word[0] for word in words_tag_vn if word[1].startswith('NN'))
    # return processed post and tags
    return vn_sent, words_tag_vn


def read_xml(blog_xml, file_name):
    try:
        parser = etree.XMLParser(recover=True)
        tree = etree.parse(blog_xml, parser)
        root = tree.getroot()
        # find all post in xml
        posts = root.findall('post')
        # print reading xml file information
        print("\tReading xml file: ", file_name)
        xml_files.append(file_name)
        post_sent = []
        post_tags = []
        for post in posts:
            s = post.text
            sent, tags = preprocess_post(s)
            post_sent.append(sent)
            post_tags.append(tags)
        return post_sent, post_tags

    except Exception as e:
        # print error format file information
        print("\tError reading: ", blog_xml, " Error message: ", e)
        # record xml files with reading error
        error_files.append(blog_xml)


def unpack_word_weight(vec, word_weight):
    """
    Given the CountVector and the fit_transformed
    word count sparse array obtain each documents'
    word count dictionary
    """
    feature_names = np.array(vec.get_feature_names())
    data = word_weight.data
    indptr = word_weight.indptr
    indices = word_weight.indices
    n_docs = word_weight.shape[0]

    word_weight_list = []
    for i in range(n_docs):
        doc = slice(indptr[i], indptr[i + 1])
        count, idx = data[doc], indices[doc]
        feature = feature_names[idx]
        word_weight_dict = Counter({k: v for k, v in zip(feature, count)})
        word_weight_list.append(word_weight_dict)
    return word_weight_list


def find_expand_topic(topic, dimension):
    """
    find expand topics in related category
    :param dimension:
    :param topic:
    :return:
    """
    expand_topics = []
    for post_tags in blog_tokens[dimension]:
        for i, word in enumerate(post_tags):
            if word[0] == topic and word[1].startswith('NN'):
                if (i - 2 >= 0) and (i + 2 < len(post_tags)):
                    expand_topic = ' '.join([t[0] for t in post_tags[i - 2:i + 3]])
                    expand_topics.append(expand_topic)
    return expand_topics


def store_data(pickle_file):
    """
    Store processed blogs and blog tags to local file
    :param pickle_file: pickle file name
    :return:
    """
    db = {'xml_files': xml_files,
          'topic_results': topic_results,
          'expand_topic_results': expand_topic_results,
          'tfidf_topic_results': tfidf_topic_results,
          'tfidf_expand_topic_results': tfidf_expand_topic_results}
    with open(pickle_file, 'wb') as f:
        pickle.dump(db, f)


def load_data(pickle_file):
    """
    Load pickle file and return data
    :param pickle_file:
    :return:
    """
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    # ############### Extract high frequency topics ###############
    ITERATION_NUM = 3
    for iteration in range(ITERATION_NUM):
        # xml file list
        xml_files = []
        # Error reading files
        error_files = []
        # All blog contents
        blogs = {"Males": [],
                 "Females": [],
                 "Age below 20": [],
                 "Age above 20": [],
                 "Everyone": []}
        # All blog tokens container: xml_file_name, [post1 tokens, post2 tokens, ...]
        blog_tokens = {"Males": [],
                       "Females": [],
                       "Age below 20": [],
                       "Age above 20": [],
                       "Everyone": []}
        # categories to store most frequent topics
        topic_results = {"Males": [],
                         "Females": [],
                         "Age below 20": [],
                         "Age above 20": [],
                         "Everyone": []}
        # categories to store most frequent topics
        expand_topic_results = {"Males": [],
                                "Females": [],
                                "Age below 20": [],
                                "Age above 20": [],
                                "Everyone": []}
        # categories to store most frequent topics
        tfidf_topic_results = {"Males": [],
                               "Females": [],
                               "Age below 20": [],
                               "Age above 20": [],
                               "Everyone": []}

        # categories to store most frequent topics
        tfidf_expand_topic_results = {"Males": [],
                                      "Females": [],
                                      "Age below 20": [],
                                      "Age above 20": [],
                                      "Everyone": []}
        # Task start information
        now = datetime.now()
        start_time = now.strftime("%D %H:%M:%S")
        time_label = now.strftime("%H-%M-%S")
        print(f"\n===== Task {iteration + 1}/{ITERATION_NUM}: Start time {start_time} =====")

        # Randomly select a number of files as input
        random_file_list = random.sample(listdir(BLOG_FOLDER), k=15)
        print("Reading posts from files ...")
        for file in random_file_list:
            # Extract information from file name
            blog_info = file.split('.')
            blogger_sex = blog_info[1].lower()
            blogger_age = int(blog_info[2])

            # Create xml full file path
            xml_path = join(BLOG_FOLDER, file)
            if isfile(xml_path):
                try:
                    # Read from xml, pre-processing and tokenize
                    sent_list, tag_list = read_xml(xml_path, file)
                    # Add tokenized blogs into related categories
                    if blogger_sex == 'male':
                        blogs["Males"].append(sent_list)
                        blog_tokens["Males"].extend(tag_list)
                    else:
                        blogs["Females"].append(sent_list)
                        blog_tokens["Females"].extend(tag_list)
                    if blogger_age <= 20:
                        blogs["Age below 20"].append(sent_list)
                        blog_tokens["Age below 20"].extend(tag_list)
                    else:
                        blogs["Age above 20"].append(sent_list)
                        blog_tokens["Age above 20"].extend(tag_list)
                    # Add blogs into everyone_category
                    blogs["Everyone"].append(sent_list)
                    blog_tokens["Everyone"].extend(tag_list)

                except Exception as e:
                    print(e)
        # ############### Extract expand topics (2verb/nouns before and 2verb/nouns) ###############
        # Search for most frequent topics
        print("\nSearching for most frequent topics ...")
        for category_name in blogs:
            corpus = []
            for blog in blogs[category_name]:
                corpus.append(' '.join(blog))

            # most frequent topic analysis
            vect = CountVectorizer()
            word_weights = vect.fit_transform(corpus)
            frequency_weights = unpack_word_weight(vect, word_weights)
            most_frequent_topics = Counter()
            for counter in frequency_weights:
                most_frequent_topics += counter
            topics = most_frequent_topics.most_common(5)
            topic_results[category_name].extend(topics)

            print(f"\nFound most frequent topics for {category_name}: \n\t", topics)
            for tpc in topics:
                ep_tpc = find_expand_topic(tpc[0], category_name)
                print(f"Expanded topics for {category_name}-{tpc[0]}: \n\t", ep_tpc[:5])
                expand_topic_results[category_name]\
                    .append(ep_tpc)

            # TF-IDF analysis
            # tf-idf instead of bag of words
            tfidf_vect = TfidfVectorizer()
            tfidf_word_weight = tfidf_vect.fit_transform(corpus)
            tfidf_weights = unpack_word_weight(tfidf_vect, tfidf_word_weight)
            high_tfidf_topics = Counter()
            for counter in tfidf_weights:
                high_tfidf_topics += counter
            tfidf_topics = high_tfidf_topics.most_common(5)
            tfidf_topic_results[category_name]\
                .extend(tfidf_topics)

            print(f"\nTFIDF: Found topics with highest TF-IDF for {category_name}: \n\t", tfidf_topics)
            for tpc in tfidf_topics:
                ep_tpc = find_expand_topic(tpc[0], category_name)
                print(f"TFIDF: Expanded topics for {category_name}-{tpc}: \n\t", ep_tpc[:5])
                tfidf_expand_topic_results[category_name]\
                    .append(ep_tpc)

        store_data('results_' + time_label + '_alldata')
        end_time = datetime.now().strftime("%D %H:%M:%S")
        print(f"===== Task {iteration + 1}/{ITERATION_NUM}: End time {end_time} =====")
