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
import re
from collections import Counter, namedtuple
import heapq
from scipy import sparse
from shutil import get_terminal_size
import sys



def timeit(method):
    def timed(*args, **kw):
        ts = time.perf_counter()
        result = method(*args, **kw)
        te = time.perf_counter()
        print('%r  %2.2f s' % (method.__name__, (te - ts)))
        return result
    return timed


class SearchEngine:
    def __init__(self, build_essentials=True, data_dir="./data/"):

        self.data_dir = data_dir
        self.proc_folder = "processed/"

        self.stemmer = EnglishStemmer()
        self.stop_words = set(stopwords.words('english'))

        self.vocab = None
        self.documents = None
        self.nr_docs = None  # no documents yet available
        self.inv_index = None  # no inverted index yet available
        self.idf = None  # no inverse document frequency yet available

        self.nltk_check_downloaded()

        if build_essentials:
            self.build_invert_idx()
        self.built = build_essentials

    @timeit
    def _create_vocab(self, docs=None):
        fname = f"{self.data_dir}vocabulary.csv"
        if not isfile(fname):
            docs = self._process_docs(docs)
            self.vocab = set()
            for doc in docs.values():
                self.vocab.update(doc)
            self.vocab = pd.DataFrame(pd.Series(np.arange(len(self.vocab)), index=self.vocab), columns=["term_id"])
            self.vocab.to_csv(fname)
        else:
            self.vocab = pd.read_csv(fname, index_col=0, header=0, keep_default_na=False, na_values=[""])

    def _load_data_by_nr(self, doc_nrs):
        docs = dict()
        for doc_nr in doc_nrs:
            docs[doc_nr] = pd.read_csv(f"{self.data_dir + self.proc_folder}doc_{doc_nr}.tsv", header=0, sep="\t")
        return docs

    def _load_data_complete(self, as_dict=True):
        docs = pd.read_csv(f"{self.data_dir}Airbnb_Texas_Rentals.csv", header=0, sep=",")
        docs.drop(columns="Unnamed: 0", inplace=True)
        if as_dict:
            ds = dict()
            for idx, doc in docs.iterrows():
                ds[idx] = doc
            docs = ds
        return docs

    def _split_data_in_ads(self):
        docs = self._load_data_complete()
        for i, row in docs.items():
            row_to_frame = row.to_frame().T
            row_to_frame.to_csv(f"{self.data_dir + self.proc_folder}doc_{i}.tsv", sep="\t", index=None)

    @staticmethod
    def nltk_check_downloaded():
        try:
            nltk.data.find('stopwords')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def _process_text(self, text):
        """
        Remove special characters and superfluous whitespaces from text body. Also
        send text to lower case, tokenize and stem the terms.
        :param text: str, the text to process.
        :return: generator, yields the processed word in iteration
        """
        text = self.doc_to_string(text).lower()
        sub_re = r"[^A-Za-z']"
        text = re.sub(sub_re, " ", text)
        for i in word_tokenize(text):
            if i not in self.stop_words:
                w = self.stemmer.stem(i)
                if len(w) > 1:
                    yield(w)

    def _process_docs(self, docs=None):
        if docs is None:
            docs = self._load_data_complete(as_dict=False)
            docs = (docs["description"] + " " + docs["title"]).to_frame()

        if isinstance(docs, pd.DataFrame):
            d_out = dict()
            for idx, doc in docs.iterrows():
                d_out[idx] = list(self._process_text(doc))
        elif isinstance(docs, dict):
            d_out = docs
            for docnr, doc in docs.items():
                d_out[docnr] = list(self._process_text(doc))
        else:
            raise ValueError("Container type has no handler.")
        return d_out

    @timeit
    def build_invert_idx(self, docs=None, read_fname="inverted_index.txt",
                         write_fname="inverted_index.txt", load_from_file=False):
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
            self._create_vocab(docs)
        file = f"{self.data_dir}{read_fname}"
        TermSet = namedtuple("TermSet", "docID tfidf")
        if isfile(file) and load_from_file:
            idf_dict = dict()
            inv_index = dict()
            with open(file, "r") as f:
                # load all the information from the file into memory
                for rowidx, line in enumerate(f):
                    if rowidx > 0:
                        term, idf_doclist = line.strip().split(":", 1)
                        idf, doclist = idf_doclist.split("|", 1)
                        idf_dict[term] = idf
                        doclist = list(map(lambda x: re.search("\d+,\s?(\d[.])?\d+", x).group().split(","),
                                           doclist.split(";")))
                        inv_index[term] = [TermSet(*list(map(float, docl))) for docl in doclist]
            # recreate the number of documents used for the inverted index
            nr_docs = set()
            for term, doclist in inv_index.items():
                nr_docs = nr_docs.union(doclist)
            self.nr_docs = len(nr_docs)
        else:
            # the final inverted index container, defaultdict, so that new terms
            # can be searched and get an empty set back
            inv_index = defaultdict(list)
            docs, idf_dict, term_freqs, doc_counters = self._build_idf(docs)
            self.nr_docs = len(doc_counters)
            for docnr, doc in docs.items():
                # weird, frequency pairs for this document
                freqs = doc_counters[docnr]
                for word, word_freq in freqs.items():
                    # nr of words in this document
                    n_terms = sum(freqs.values())
                    # store which document and frequency
                    inv_index[word].append(TermSet(docnr, word_freq / n_terms * idf_dict[word]))
            # write the built index to file
            with open(f"{self.data_dir}{write_fname}", "w") as f:
                f.write("Word: [Documents list]\n")
                for word, docs in inv_index.items():
                    docs = [(doc.docID, doc.tfidf) for doc in docs]
                    f.write(f"{word}: {idf_dict[word]} | {';'.join([str(doc) for doc in docs])}\n")
        self.inv_index = inv_index
        self.idf = idf_dict
        return inv_index

    @timeit
    def _build_idf(self, docs=None):
        docs = self._process_docs(docs)
        nr_docs = len(docs)
        idf = defaultdict(lambda: np.math.log(len(docs) + 1))
        # dict to track nr of occurences of each term
        term_freqs = defaultdict(int)
        # dict to store counter of words in docs
        doc_counters = dict()
        for docnr, doc in docs.items():
            freqs = Counter(doc)
            doc_counters[docnr] = freqs
            for word in freqs.keys():
                term_freqs[word] += 1
        for word in self.vocab.index:
            # nr of documents with this term in it
            nr_d_with_term = term_freqs[word]
            # inverse document frequency for this term and this document
            idf[word] = np.math.log((float(nr_docs + 1) / (1 + nr_d_with_term)))
        self.idf = idf
        return docs, idf, term_freqs, doc_counters

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

    def tfidf_query(self, query):
        """
        TF-IDF weigh a given query vector
        :param query: ndarray, list or similar; query terms in vector for (terms are strings)
        :return: dict, dictionary with term and weights associated to it
        """

        n_terms = len(query)
        query_terms = Counter(query)

        # drop all unseen before words, tfidf weight the rest
        query_dic = dict()
        for term, freq in query_terms.items():
            query_dic[term] = freq / n_terms * self.idf[term]

        return query_dic

    @timeit
    def _process_query_conjunctive(self, query):
        query_proc = list(self._process_text(query))
        # Iterate over inverted index find which documents have the terms of the search.
        # Iterate over the query terms by popping element 0 and cut the set of documents
        # containing this term with the current set of candidates.
        # In math: INTERSECT_{term in Vocab} {d in Documents: term in d}
        docs_to_search = set(np.arange(self.nr_docs))  # initialize with all the doc numbers
        query_terms = set(query_proc)  # the list to pop terms from
        while query_terms:
            term = query_terms.pop()
            if term in self.inv_index:
                # get the documents that contain this term
                ds = [doc for doc, f in self.inv_index[term]]
                # intersect all the relev docs in this set with the candidates
                docs_to_search = docs_to_search.intersection(ds)
            else:
                # term isn't in vocabulary -> no document contains this term
                # return no document as conjunctive query was unsuccessful
                return query, None

        if not docs_to_search:  # equiv to empty set
            return None  # no document contained all terms

        rel_docs = self._load_data_by_nr(docs_to_search)

        return query, rel_docs

    @timeit
    def _process_query_cosinesim(self, query, top_k=10):
        _, rel_docs = self._process_query_conjunctive(query)
        if rel_docs is not None:
            query_proc = list(self._process_text(query))
            query_dic = self.tfidf_query(query_proc)
            col = []
            row = []
            data = []
            for d_nr, content in rel_docs.items():
                content = set(self._process_text(content["title"] + " " + content["description"]))
                for term in content:
                    col.append(self.vocab.loc[term, "term_id"])
                    row.append(d_nr)
                    for termset in self.inv_index[term]:
                        if termset.docID == d_nr:
                            data.append(termset.tfidf)
                            break
            shape = self.nr_docs, len(self.vocab)
            docs_repr = sparse.csr_matrix((data, (row, col)), shape=shape, dtype=float)

            col = []
            row = []
            data = []
            for term, tfidf_weight in query_dic.items():
                col.append(self.vocab.loc[term, "term_id"])
                row.append(0)
                data.append(tfidf_weight)
            shape = 1, len(self.vocab)
            query_rep = sparse.csr_matrix((data, (row, col)), shape=shape, dtype=float)

            similarities = cosine_similarity(docs_repr, query_rep)
            # load the documents to gain the top ones
            docs = self._load_data_complete(as_dict=False)
            docs.loc[:, "score"] = similarities
            # create heap for the top documents
            top_heap = [(sim, docnr) for sim, docnr in zip(-similarities[:, 0], range(docs.shape[0]))]
            heapq.heapify(top_heap)
            top_k_docs = []
            for i in range(top_k):
                try:
                    sim, docnr = heapq.heappop(top_heap)
                    top_k_docs.append(docs.loc[docnr])
                except IndexError:
                    break  # no more elements in the heap
            return query, top_k_docs
        else:
            return query, None

    def search_cosine(self):
        print("Please enter search query: ", end=" ")
        query = input()
        print('searching...')
        _, top_k_docs = self._process_query_cosinesim(query, top_k=10)
        print("Search finished."), sys.stdout.flush()
        if top_k_docs is not None:
            rel_cols = ["title", "description", "city", "url", "score"]
            docs = {rank+1: doc[rel_cols] for (rank, doc) in enumerate(top_k_docs)}
            self._print_search_res(query, docs)
        else:
            print("No announcement matched the search.")
        return

    def search_conjunctive(self):
        print("Please enter search query: ", end=" ")
        query = "cosy bedroom"
        print('searching...')
        _, all_rel_docs = self._process_query_conjunctive(query)
        print("Search finished."), sys.stdout.flush()
        if all_rel_docs is not None:
            rel_cols = ["title", "description", "city", "url"]
            docs = {key: doc.loc[0, rel_cols] for (key, doc) in all_rel_docs.items()}
            self._print_search_res(query, docs, has_score=False)
        else:
            print("No announcement matched the search.")
        return

    @staticmethod
    def _print_search_res(query, docs, has_score=True):
        t_size = get_terminal_size().columns
        t_size_capped = t_size - 10

        # calculate the column widths for all entry, given the percentages
        # and terminal size (if terminal is too small, this will break maybe)
        wd = {"t": .2, "d": .3, "c": .15, "u": .25}

        if has_score:
            for col in wd:
                wd[col] -= .01
            wd["r"] = 7/100
            wd["s"] = 1 - sum(wd.values())

        for col in wd:
            wd[col] = int(wd[col] * t_size_capped)

        if has_score:
            print_widths = [wd[k] for k in ("r", "t", "d", "c", "u", "s")]
        else:
            print_widths = [wd[k] for k in ("t", "d", "c", "u")]

        print(f"\nSearch query: {query if len(query) <= 100 else query[0:100] + '...'}")
        search_res_msg = f"Seach results:\n\n"
        search_res_middle = f"{'Title'.center(wd['t'], ' ')}|" \
                            f"{'Description'.center(wd['d'], ' ')}|"\
                            f"{'City'.center(wd['c'], ' ')}|" \
                            f"{'Url'.center(wd['u'], ' ')}"
        if has_score:
            search_res_msg += f"{'Rank'.center(wd['r'], ' ')}|" + search_res_middle
            search_res_msg += f"|{'Score'.center(wd['s'], ' ')}"
            ResArray = namedtuple("ResArray", "rank title desc city url score")
        else:
            search_res_msg += search_res_middle
            ResArray = namedtuple("ResArray", "title desc city url")
        print(search_res_msg)

        def fit(string, s_list, max_width):
            while True:
                try:
                    next_ = s_list[0]
                except IndexError:  # list is now empty
                    break
                if len(f"{string} {next_}") <= max_width-2:
                    next_ = s_list.pop(0)
                    string += " " + next_
                else:
                    break
            return string

        max_nr_rows_per_res = 5
        for rank, doc in docs.items():
            # ad separation lines
            print(*(["="] * t_size), sep="", end="")
            if has_score:
                title, desc, city, url, score = doc
            else:
                title, desc, city, url = doc
            title, desc, city = list(map(lambda x: x.replace("\\n", "").split(" "), (title, desc, city)))
            # array for holding the prepared rows to print
            print_res = []
            for r in range(max_nr_rows_per_res):
                t, d, c = list(map(fit, [""]*3, (title, desc, city), [wd[k] for k in ("t", "d", "c")]))
                try:
                    u = url[r*wd["u"]:(r+1)*wd["u"]-2]
                except IndexError:  # index error, url already covered by prev row
                    u = ""

                if has_score:
                    if r == 0:
                        s = str(round(score, 3))
                        rank = str(rank)
                    else:
                        rank, s = "", ""

                    print_res.append(ResArray(rank, t, d, c, u, s))
                else:
                    print_res.append(ResArray(t, d, c, u))

            if len(title) > 0:  # title wasnt fully covered
                var = list(print_res[-1].title)  # string of title to list of chars
                var[-3:] = "..."  # last three chars of title
                t = "".join(var)
            if len(desc) > 0:  # desc wasnt fully covered
                var = list(print_res[-1].desc)
                var[-3:] = "..."
                d = "".join(var)
            if len(city) > 0:  # city wasnt fully covered
                var = list(print_res[-1].city)
                var[-3:] = "..."
                c = "".join(var)
            if len(url[max_nr_rows_per_res*wd["u"]:]):  # still url left
                var = list(print_res[-1].url)  # string of title to list of chars
                var[-3:] = "..."  # last three chars of title
                u = "".join(var)
            # reassign last row
            print_res[-1] = ResArray(rank, t, d, c, u, print_res[-1].score) if has_score \
                       else ResArray(t, d, c, u)
            for row in print_res:
                print(*map(lambda s, l: s.center(l, " "), row, print_widths), sep="|")

        print(*(["="] * t_size), sep="")
        return


if __name__ == '__main__':
    se = SearchEngine(build_essentials=True)
    se.search_conjunctive()
    se.search_cosine()





