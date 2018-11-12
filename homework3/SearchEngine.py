import nltk
from os.path import isfile
from nltk import word_tokenize
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import time
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
import re
from collections import Counter, OrderedDict, namedtuple


def timeit(method):
    def timed(*args, **kw):
        ts = time.clock()
        result = method(*args, **kw)
        te = time.clock()
        print('%r  %2.2f s' % (method.__name__, (te - ts)))
        return result
    return timed


class SearchEngine:
    def __init__(self, data_dir="./data/"):
        self.data_dir = data_dir
        self.proc_folder = "processed/"
        self.vocab = set()
        self.stemmer = EnglishStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.documents = None
        self.nr_docs = None  # no documents yet available
        self.inv_index = None  # no inverted index yet available
        self.idf = None  # no inverse document frequency yet available
        # self.stop_words.update(['.', ',', '"', '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
        self.nltk_check_downloaded()
        self.vocab = None
        self._create_vocab()

    @timeit
    def _create_vocab(self):
        fname = f"{self.data_dir}vocabulary.csv"
        if not isfile(fname):
            docs = self._load_data_complete()
            words = (docs["description"] + " " + docs["title"]).to_frame().rename(columns={0: "ad"})
            words = words.apply(lambda x: list(self.process_text(x)), axis=1)
            self.documents = words
            self.vocab = set()
            for i, doc in words.iteritems():
                self.vocab.update(doc)
            self.vocab = pd.DataFrame(np.arange(len(self.vocab)), index=self.vocab, columns=["term_id"])
            self.vocab.write_csv(fname)
        else:
            self.vocab = pd.read_csv(fname, index=0)

    def _load_data_docwise(self):
        i = 0
        docs = []
        while True:
            try:
                doc = pd.read_csv(f"{self.data_dir + self.proc_folder}doc_{i}.tsv", header=0, sep="\t")
            except EmptyDataError:
                break
            docs.append(doc)
            i += 1
        self.documents = pd.concat(docs)
        return self.documents

    def _load_data_complete(self):
        airbnb = pd.read_csv(f"{self.data_dir}Airbnb_Texas_Rentals.csv", header=0, sep=",")
        airbnb.drop(columns="Unnamed: 0", inplace=True)
        return airbnb

    def _prep_data(self):
        airbnb = self._load_data_complete()
        for i, row in airbnb.iterrows():
            row_to_frame = row.to_frame().T
            row_to_frame.to_csv(f"{self.data_dir + self.proc_folder}doc_{i}.tsv", sep="\t", index=None)

    @staticmethod
    def nltk_check_downloaded():
        try:
            nltk.download('stopwords')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def process_text(self, text):
        """
        Remove special characters and superfluous whitespaces from text body. Also
        send text to lower case, tokenize and stem the terms.
        :param text: str, the text to process.
        :return: generator, yields the processed word in iteration
        """
        text = self.doc_to_string(text).lower()
        # sub_re = r"([\\!\"§$€%&/()=?*+\'#_\-.:,;\d])|(\s{2,})"
        sub_re = r"([^A-Za-z'])|(\s)"
        text = re.sub(sub_re, " ", text)
        for i in word_tokenize(text):
            if i not in self.stop_words:
                w = self.stemmer.stem(i)
                if len(w) > 1:
                    yield(w)

    @timeit
    def build_invert_idx(self, docs, read_fname="inverted_index.txt",
                         write_fname="inverted_index.txt"):
        """
        Build the inverted index for the terms in a collection of documents. Will load a
        previously build inverted index from file if it detects the file existing.
        :param docs: list, collection of documents of type list, tuple or ndarray
        :param read_fname: str, filename of the inverted txt to load. Needs to be built in the
                              specified way of the method
        :param write_fname: str, filename to write the inverted index to.
        :return: dict, the inverted index with terms as keys and (doc_nr, relative_term_frequency)
                       as values.
        """
        if self.vocab is None:
            self._create_vocab()
        file = f"{self.data_dir}{read_fname}"
        if isfile(file):
            TermSet = namedtuple("TermSet", "id idf tfidf_id")
            inv_index = dict()
            with open(file, "r") as f:
                # load all the information from the file into memory
                for rowidx, line in enumerate(f):
                    if rowidx > 0:
                        term, doclist = line.split(":")
                        doclist = list(map(lambda x: re.search("\d+,\s?(\d[.])?\d+", x).group().split(","),
                                           doclist.split(";")))
                        doclist = [TermSet(map(float, doc)) for doc in doclist]
                        inv_index[term.strip()] = doclist
            # recreate the number of documents used for the inverted index
            nr_docs = set()
            for term, doclist in inv_index.items():
                nr_docs = nr_docs.union(doclist)
            self.nr_docs = len(nr_docs)
        else:
            if self.documents is None:
                docs = [list(self.process_text(doc)) for doc in docs]
            else:
                docs = self.documents
            self.nr_docs = len(docs)
            # the final inverted index container, defaultdict, so that new terms
            # can be searched and get an empty set back
            inv_index = defaultdict(list)
            idf, term_freqs, doc_counters = self._build_idf(docs, processed=True)
            for docnr, doc in enumerate(docs):
                # weird, frequency pairs for this document
                freqs = doc_counters[docnr]
                for word, word_freq in freqs.items():
                    # nr of documents with this term in it
                    nr_d_with_term = term_freqs[word]
                    # nr of words in this document
                    n_terms = sum(freqs.values())
                    # inverse document frequency for this term and this document
                    idf = np.math.log((float(self.nr_docs) + 1) / (1 + nr_d_with_term))
                    # store which document and frequency
                    inv_index[word].append((docnr, idf, word_freq / n_terms * idf))
            # write the built index to file
            with open(f"{self.data_dir}{write_fname}", "w") as f:
                f.write("Word: [Documents list]\n")
                for word, docs in inv_index.items():
                    f.write(f"{word}: {';'.join([str(doc) for doc in docs])}\n")
        self.inv_index = inv_index
        return inv_index

    @timeit
    def _build_idf(self, docs, processed=False):
        if not processed:
            docs = [list(self.process_text(doc)) for doc in docs]

        idf = defaultdict(lambda: np.math.log(len(docs) + 1))
        # dict to track nr of occurences of each term
        term_freqs = defaultdict(int)
        # dict to store counter of words in docs
        doc_counters = dict()
        for docnr, doc in enumerate(docs):
            freqs = Counter(doc)
            doc_counters[docnr] = freqs
            for word in freqs.keys():
                term_freqs[word] += 1
        for word in self.vocab:
            # nr of documents with this term in it
            nr_d_with_term = term_freqs[word]
            # inverse document frequency for this term and this document
            idf[word] = np.math.log((float(self.nr_docs) + 1) / (1 + nr_d_with_term))
        self.idf = idf
        return idf, term_freqs, doc_counters

    @staticmethod
    def doc_to_string(doc):
        """
        Converts a document to a string. Can take a list, ndarray, tuple to convert to str
        :param doc: iterable, container of the document
        :return: str, the to string converted document.
        """
        if isinstance(doc, str):
            return doc
        elif isinstance(doc, np.ndarray):
            doc = " ".join(list(map(str, doc.flatten())))
        elif isinstance(doc, (list, tuple)):
            doc = " ".join(sum(doc, []))  # reduces [["1", "2], ["3"]] to ["1", "2", "3"]
        elif isinstance(doc, (pd.DataFrame, pd.Series)):
            doc = " ".join(list(map(str, doc.values.flatten())))
        else:
            raise ValueError("Can't convert file type of document to string.")
        return doc

    @timeit
    def tfidf_query(self, query):
        """
        TF-IDF weigh a given query vector
        :param query: ndarray, list or similar; query terms in vector for (terms are strings)
        :return: dict, dictionary with term and weights associated to it
        """

        n_terms = len(query)
        query_terms = Counter(query)

        # drop all unseen before words, tfidf weight the rest
        query_dic = OrderedDict()
        for idx, term in enumerate(query_terms):
            query_dic[term] = query_terms[term] / n_terms * self.inv_index[term].idf

        return query_dic

    def search_query(self, query):
        query_proc = list(self.process_text(query))
        query_dic = self.tfidf_query(query_proc)
        nr_terms = len(self.vocab)
        query_rep = np.array([0] * nr_terms, dtype="float32").reshape(1, -1)
        # we have an OrderedDict from the inverted index
        # so we can iterate over the items and place the vector representation
        # by the index of the term in the loop
        docs_to_search = set()
        for term in query_terms:
            if term in self.inv_index:
                ds = [d for d, f in self.inv_index[term]]
                docs_to_search = docs_to_search.union(ds)
        # initialize the document frame for which we are computing the similarities
        doc_repr = pd.DataFrame(0, index=docs_to_search, columns=all_terms)
        for doc in doc_repr.index:
            # we now reconstruct the documents words and associated frequencies
            # by iterating over the inverted index and checking for entries with doc
            for idx, term in enumerate(all_terms):
                # only change weights of terms that are used in the index, unseen terms = 0 !
                if term in self.inv_index:
                    # get frquency of the term corresponding to the doc
                    freq = [f for d, f in self.inv_index[term] if d == doc]
                    if freq:
                        # if this term is used in the doc then add its weight
                        doc_repr.loc[doc, term] = freq[0] * self.idf[term]
                # only if the term is in the query, we add its weight
                if term in query_dic:
                    query_rep[0, idx] = query_dic[term]

        similarities = cosine_similarity(doc_repr, query_rep)
        return query, similarities

if __name__ == '__main__':

    se = SearchEngine()

