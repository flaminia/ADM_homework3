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
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.extra import rate_limiter
from geopy import distance
import socket


def timeit(method):
    def timed(*args, **kw):
        ts = time.perf_counter()
        result = method(*args, **kw)
        te = time.perf_counter()
        print('%r  %2.2f s' % (method.__name__, (te - ts)))
        return result
    return timed


def internet(host="8.8.8.8", port=53, timeout=3):
    """
    Host: 8.8.8.8 (google-public-dns-a.google.com)
    OpenPort: 53/tcp
    Service: domain (DNS/TCP)
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception as ex:
        print(ex.message)
        return False


class SearchEngine:
    def __init__(self, build_essentials=True, data_dir="./data/"):

        self.data_dir = data_dir
        self.proc_folder = "processed/"

        self.stemmer = EnglishStemmer()
        self.stop_words = set(stopwords.words('english'))

        self.vocab = None
        self.documents = None
        self.inv_index = None  # no inverted index yet available
        self.idf = None  # no inverse document frequency yet available

        self.nltk_check_downloaded()

        # compute statistics of the data set
        # (if the dataset was much larger this would need to be a continuously
        # updated data file to read from, and periodically update)
        docs = self._load_data_complete(as_dict=False)
        self.nr_docs = docs.shape[0]
        self.doc_nrs = docs.index
        self.max_rate = docs["average_rate_per_night"].apply(lambda x: int(x[1:])).max()
        self.max_nr_bedrooms = docs["bedrooms_count"].max()
        self.most_recent = docs["date_of_listing"].apply(lambda x: datetime.strptime(x, "%B %Y")).max()
        self.available_cities = pd.unique(docs["city"])

        if build_essentials:
            docs_text_parts = docs["title"] + " " + docs["description"]
            self.build_invert_idx(docs_text_parts)
        self.built = build_essentials

    @timeit
    def _create_vocab(self, docs=None):
        fname = f"{self.data_dir}vocabulary.csv"
        if not isfile(fname):
            docs = self._process_docs(docs)
            self.vocab = set()
            for doc in docs.values():
                self.vocab.update(doc)
            self.vocab = pd.DataFrame(pd.Series(np.arange(len(self.vocab)), index=self.vocab),
                                      columns=["term_id"])
            self.vocab.to_csv(fname)
        else:
            self.vocab = pd.read_csv(fname, index_col=0, header=0,
                                     keep_default_na=False, na_values=[""])

    def _load_data_by_nr(self, doc_nrs):
        docs = dict()
        for doc_nr in doc_nrs:
            docs[doc_nr] = pd.read_csv(f"{self.data_dir + self.proc_folder}doc_{doc_nr}.tsv",
                                       header=0, sep="\t").set_index([[doc_nr]])
        return docs

    def _load_data_complete(self, as_dict=True):
        docs = pd.read_csv(f"{self.data_dir}Airbnb_Texas_Rentals.csv", header=0, sep=",")
        docs.drop(columns="Unnamed: 0", inplace=True)
        docs = docs.dropna().drop_duplicates(subset=["title", "description", "city"])
        docs.reset_index(drop=True, inplace=True)
        if as_dict:
            ds = dict()
            for docnr, doc in docs.iterrows():
                ds[docnr] = doc
            docs = ds
        return docs

    def _split_data_in_ads(self):
        docs = self._load_data_complete(as_dict=False)
        for i, row in docs.iterrows():
            row_to_frame = row.to_frame().T
            row_to_frame.to_csv(
                f"{self.data_dir + self.proc_folder}doc_{i}.tsv",
                sep="\t", index=None
            )

    @staticmethod
    def nltk_check_downloaded():
        try:
            nltk.data.find('corpora/stopwords')
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
            docs_generator = docs.iterrows()
        elif isinstance(docs, pd.Series):
            docs_generator = docs.iteritems()
        elif isinstance(docs, dict):
            docs_generator = docs.items()
        else:
            raise ValueError("Container type has no handler.")

        d_out = dict()
        for docnr, doc in docs_generator:
            d_out[docnr] = list(self._process_text(doc))
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
        else:
            # the final inverted index container, defaultdict, so that new terms
            # can be searched and get an empty list back
            inv_index = defaultdict(list)
            docs, idf_dict, term_freqs, doc_counters = self._build_idf(docs)
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
    def _process_query_rel_docs(self, query, conjunctive=True):
        query_proc = list(self._process_text(query))
        if conjunctive:
            cand_update = lambda set1, set2: set1.intersection(set2)
            docs_to_search = set(self.doc_nrs)  # initialize with all the doc numbers
        else:
            cand_update = lambda set1, set2: set1.union(set2)
            docs_to_search = set()  # initialize with empty set
        # Iterate over inverted index find which documents have the terms of the search.
        # Iterate over the query terms by popping element 0 and cut (unite) the set of documents
        # containing this term with the current set of candidates. In math terminology:
        # INTERSECT_{term in Vocab} {d in Documents: term in d} or
        # UNION_{term in Vocab} {d in Documents: term in d}
        query_terms = set(query_proc)  # the list to pop terms from
        while query_terms:
            term = query_terms.pop()
            if term in self.inv_index:
                # get the documents that contain this term
                ds = [doc for doc, f in self.inv_index[term]]
                # intersect all the relev docs in this set with the candidates
                docs_to_search = cand_update(docs_to_search, ds)
            else:
                # term isn't in vocabulary -> no document contains this term
                # return no document as conjunctive query was unsuccessful
                return query, None

        if not docs_to_search:  # equiv to empty set
            return query, None  # no document contained all terms

        rel_docs = self._load_data_by_nr(docs_to_search)

        return query, rel_docs

    @timeit
    def _process_query_cosinesim(self, query, top_k=10, conjunctive=True):
        _, rel_docs = self._process_query_rel_docs(query, conjunctive)
        if rel_docs is not None:
            _, similarities = self._compute_cosine(query, rel_docs)
            # create heap for the top documents
            top_heap = [(sim, docnr) for sim, docnr in zip(-similarities[:, 0], range(self.nr_docs))]
            heapq.heapify(top_heap)
            top_k_docs = []
            for i in range(top_k):
                try:
                    sim, docnr = heapq.heappop(top_heap)
                    doc = rel_docs[docnr]
                    docnr["score"] = -sim
                    top_k_docs.append(doc)
                except IndexError:
                    break  # no more elements in the heap
            return query, top_k_docs
        else:
            return query, None

    def _compute_cosine(self, query, docs):
        query_proc = list(self._process_text(query))
        query_dic = self.tfidf_query(query_proc)
        col = []
        row = []
        data = []
        for d_nr, content in docs.items():
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
        return query, similarities

    def _process_query_happy_score(self, query, info, top_k=10):

        # get all documents, that have at least one word in common with search
        _, rel_docs = self._process_query_rel_docs(query, conjunctive=False)

        # convert dict to DataFrame for easier handling
        d = pd.concat((doc for docnr, doc in rel_docs.items()))
        # prepare frame for calculations
        d["rate"] = d["average_rate_per_night"].apply(lambda x: float(x[1:]))
        d["date"] = d["date_of_listing"].apply(
            lambda x: datetime.strptime(x, "%B %Y")
        )
        d["date_diff"] = d["date"].apply(
            lambda x: (x - info["date_of_listing"]).total_seconds()
        )
        d.loc[d["bedrooms_count"] == "Studio", "bedrooms_count"] = 0
        d["bedrooms_count"] = d["bedrooms_count"].astype(int)

        # filter documents by provided preferences first.
        cities = info["city"]
        max_rate = info["max_rate"]
        bed_count = info["bedrooms_count"]
        date_listing = info["date_of_listing"]
        d = d[(d["city"].isin(cities)) &
              (d["rate"] <= max_rate) &
              (d["bedrooms_count"] >= bed_count) &
              (d["date_diff"] >= 0)]
        if d.empty:
            return query, None

        # create new score out of the interesting documents
        info_weights = {"cosine": .1, "distance": 0.4, "rate": .3, "date": .2}
        doc_eval = pd.DataFrame(0, columns=info_weights.keys(), index=d.index)

        online = internet()  # check if internet connection exists
        if online:
            geo_locator = Nominatim(user_agent="loc")
            # Nominatim allows only 1 request per second, no heavy uses
            geocode = rate_limiter.RateLimiter(geo_locator.geocode, min_delay_seconds=1)
            city_coords = dict()
            for city in cities:
                loc = geocode(city + ", Texas, USA")
                city_coords[city] = (loc.latitude, loc.longitude)

            # distance calculation for the city names, only works with internet connection
            def dist(x):
                city = x["city"]
                long, lat = x["longitude"], x["latitude"]
                dis = distance.distance(city_coords[city], (lat, long)).miles * 1.60934  # conv to km
                return dis

            d["distance"] = d.loc[:, ["city", "longitude", "latitude"]].apply(dist, axis=1)
            max_dist = 10  # kilometres, eval docs on this scale
            doc_eval["distance"] = 1 - d["distance"].apply(lambda x: min(x / max_dist, 1))

        else:  # no internet, let go of the distance weighting
            info_weights["distance"] = 0  # for renormalization
            doc_eval["distance"] = 0

        # normalize the weights
        norm = sum(info_weights.values())
        for weight in info_weights:
            info_weights[weight] /= norm

        # goodness of the rate
        doc_eval["rate"] = (1 - d["rate"] / max_rate)
        longest_time_period = (self.most_recent - date_listing).total_seconds()

        # goodness of the date
        doc_eval["date"] = d["date"].apply(
            lambda x: abs((x - self.most_recent).total_seconds()) / longest_time_period
        )

        _, sims = self._compute_cosine(query, {doc.name: doc for _, doc in d.iterrows()})
        # goodness of the cosine similarity
        doc_eval["cosine"] = sims[[doc.name for _, doc in d.iterrows()], :]

        # weight all the columns
        for name, weight in info_weights.items():
            doc_eval[name] = doc_eval[name] * weight

        # sum up all the different categories for an overall score, drop rest
        doc_eval = doc_eval.sum(axis=1)

        # heap always returns minimum, but we want max score -> negate numbers
        heap_list = [(-score, doc_nr) for doc_nr, score in doc_eval.iteritems()]
        heapq.heapify(heap_list)
        top_k_docs = []
        for i in range(top_k):
            try:
                sim, docnr = heapq.heappop(heap_list)
                doc = rel_docs[docnr]
                doc["score"] = -sim
                top_k_docs.append(doc)
            except IndexError:
                break  # no more elements in the heap

        return query, top_k_docs

    def search_cosine(self):
        print("Please enter search query: ", end=" ")
        query = input()
        print('searching...')
        _, top_k_docs = self._process_query_cosinesim(query, top_k=10)
        print("Search finished."), sys.stdout.flush()
        if top_k_docs is not None:
            rel_cols = ["title", "description", "city", "url", "score"]
            docs = {doc.index[0]: doc.loc[doc.index[0], rel_cols] for doc in top_k_docs}
            self._print_search_res(query, docs)
        else:
            print("No announcement matched the search.")
            docs = None
        return docs

    def search_conjunctive(self):
        print("Please enter search query: ", end=" ")
        query = input()
        print('searching...')
        _, all_rel_docs = self._process_query_rel_docs(query, conjunctive=True)
        print("Search finished."), sys.stdout.flush()
        if all_rel_docs is not None:
            rel_cols = ["doc_nr", "title", "description", "city", "url"]
            docs = {key: doc.loc[0, rel_cols] for (key, doc) in all_rel_docs.items()}
            self._print_search_res(query, docs, has_score=False)
        else:
            print("No announcement matched the search.")
            docs = None
        return docs

    def search_happy_score(self):
        null_val = None
        print("Please enter search query: ", end=" ")
        query = "bedroom" # input()
        information = dict()
        print("Optionally you may provide the following information in the order named (Separated by return key):\n")
        print("Cities (space separated):", end=" ")
        c = "Houston San Antonio" #input()
        information["city"] = c.split(" ") if c is not "" else null_val
        print("Maximum rate per night:", end=" ")
        r = 50 #input()
        information["max_rate"] = float(r) if r is not "" else null_val
        print("Minimum number of bedrooms:", end=" ")
        b = 1 #input()
        information["bedrooms_count"] = int(b) if b is not "" else null_val
        print("Date of listing after (MM/YY): ", end=" ")
        d = "01/13" #input()
        information["date_of_listing"] = datetime.strptime(d, "%m/%y") if d is not "" else null_val

        print('Search initialized...')
        _, top_k_docs = self._process_query_happy_score(query, information)
        print("Search finished."), sys.stdout.flush()
        if top_k_docs is not None:
            rel_cols = ["title", "description", "city", "url", "score"]
            docs = {doc.index[0]: doc.loc[doc.index[0], rel_cols] for doc in top_k_docs}
            self._print_search_res(query, docs, has_score=True)
        else:
            print("No announcement matched the search.")
        return top_k_docs

    @staticmethod
    def _print_search_res(query, docs, has_score=True):
        t_size = get_terminal_size().columns
        t_size_capped = t_size - 10

        # calculate the column widths for all entry, given the percentages
        # and terminal size (if terminal is too small, this will break maybe)
        wd = {"r": .07, "t": .2, "d": .3, "c": .15, "u": .25}

        if has_score:
            for col in wd:
                wd[col] -= .01
            wd["s"] = 1 - sum(wd.values())

        for col in wd:
            wd[col] = int(wd[col] * t_size_capped)

        if has_score:
            print_widths = [wd[k] for k in ("r", "t", "d", "c", "u", "s")]
        else:
            print_widths = [wd[k] for k in ("r", "t", "d", "c", "u")]

        print(f"\nSearch query: {query if len(query) <= 100 else query[0:100] + '...'}")
        search_res_msg = f"Seach results:\n\n" \
                         f"{'Doc-Nr'.center(wd['r'], ' ')}|" \
                         f"{'Title'.center(wd['t'], ' ')}|" \
                         f"{'Description'.center(wd['d'], ' ')}|"\
                         f"{'City'.center(wd['c'], ' ')}|" \
                         f"{'Url'.center(wd['u'], ' ')}"
        if has_score:
            search_res_msg += f"|{'Score'.center(wd['s'], ' ')}"
            ResArray = namedtuple("ResArray", "docnr title desc city url score")
        else:
            ResArray = namedtuple("ResArray", "docnr title desc city url")
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
        for docnr, doc in docs.items():
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

                if r == 0:
                    s = str(round(float(score), 3)) if has_score else ""
                    docnr = str(docnr)
                else:
                    docnr, s = "", ""

                arr = (docnr, t, d, c, u, s) if has_score else (docnr, t, d, c, u)
                print_res.append(ResArray(*arr))

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
            print_res[-1] = ResArray(docnr, t, d, c, u, print_res[-1].score) if has_score \
                       else ResArray(docnr, t, d, c, u)
            for row in print_res:
                print(*map(lambda string, l: string.center(l, " "), row, print_widths), sep="|")

        print(*(["="] * t_size), sep="")
        return


if __name__ == '__main__':
    se = SearchEngine(build_essentials=True)
    #se._split_data_in_ads()
    se.search_happy_score()
    #se.search_conjunctive()
    #se.search_cosine()





