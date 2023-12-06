import os
import pickle
import contextlib
import heapq
import math

from retriever.index import InvertedIndexReader, InvertedIndexWriter
from retriever.util import IdMap, merge_and_sort_posts_and_tfs
from retriever.compression import VBEPostings
from tqdm import tqdm

from nltk.stem import PorterStemmer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from operator import itemgetter
import re
from collections import defaultdict


class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """

    def __init__(self, data_dir, output_dir, postings_encoding, index_name="main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        self.doc_length = {} # Initialize doc_length as an empty dictionary

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""
        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)
        with open(os.path.join(self.output_dir, 'doc_length.dict'), 'wb') as f:
            pickle.dump(self.doc_length, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""
        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'doc_length.dict'), 'rb') as f:
            self.doc_length = pickle.load(f)

    def pre_processing_text(self, content):
        """
        Melakukan preprocessing pada text, yakni stemming dan removing stopwords
        """
        # https://github.com/ariaghora/mpstemmer/tree/master/mpstemmer

        content = content.lower()  # Mengubah teks menjadi lowercase
        stemmer = PorterStemmer()
        stemmed = stemmer.stem(content)
        remover = StopWordRemoverFactory().create_stop_word_remover()
        final = remover.remove(stemmed)
        return final

    def parsing_block(self, block_path):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk stemming bahasa Indonesia, seperti
        MpStemmer: https://github.com/ariaghora/mpstemmer 
        Jangan gunakan PySastrawi untuk stemming karena kode yang tidak efisien dan lambat.

        JANGAN LUPA BUANG STOPWORDS! Kalian dapat menggunakan PySastrawi 
        untuk menghapus stopword atau menggunakan sumber lain seperti:
        - Satya (https://github.com/datascienceid/stopwords-bahasa-indonesia)
        - Tala (https://github.com/masdevid/ID-Stopwords)

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_path : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parsing_block(...).
        """
        stemmer = PorterStemmer()
        remover = StopWordRemoverFactory().create_stop_word_remover()
        stop_factory = StopWordRemoverFactory()
        stop_words_set = set(stop_factory.get_stop_words())

        td_pairs = []
        block_dir_path = os.path.join(self.data_dir, block_path)
        for doc_name in os.listdir(block_dir_path):
            doc_path = os.path.join(block_dir_path, doc_name)
            
            doc_id = self.doc_id_map[os.path.join(block_path, doc_name)]
            
            with open(doc_path, 'rb') as file:
                try:
                    content = file.read().decode()
                except UnicodeDecodeError:
                    document = file.read().decode('latin-1')
                
                # Tokenisasi
                tokens = re.findall(r'\b\w+\b', content.lower())
                
                document_no_stopwords = []
                
                # Proses setiap token
                for token in tokens:
                    if token not in stop_words_set:
                        stemmed_token = stemmer.stem(token.lower())
                        document_no_stopwords.append(stemmed_token)

                for stemmed_token in document_no_stopwords:  
                    term_id = self.term_id_map[stemmed_token]
                    if term_id is not None:  # Pastikan token ada dalam term_id_map
                        td_pairs.append((term_id, doc_id))

        return td_pairs

    def write_to_index(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-maintain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan strategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        term_to_docs_tf = defaultdict(lambda: defaultdict(int))
        for term_id, doc_id in td_pairs:
            term_to_docs_tf[term_id][doc_id] += 1

        # Sort terms by their term IDs
        for term_id in sorted(term_to_docs_tf.keys()):
            docs_tf = term_to_docs_tf[term_id]
            # Sort doc IDs for each term
            postings_list = sorted(docs_tf.keys())
            tf_list = [docs_tf[doc_id] for doc_id in postings_list]
            index.append(term_id, postings_list, tf_list)

    def merge_index(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi merge_and_sort_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        curr, postings, tf_list = next(merged_iter)  # first item
        for t, postings_, tf_list_ in merged_iter:  # from the second item
            if t == curr:
                zip_p_tf = merge_and_sort_posts_and_tfs(list(zip(postings, tf_list)),
                                                        list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)


    def get_postings_list(self, term):
        term_id = self.term_id_map[term]
        if term_id is None:
            return [], []
        
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as index_reader:
            postings_list, tf_list = index_reader.get_postings_list(term_id)
            
        return postings_list, tf_list



    def retrieve_tfidf(self, query, k=10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        # Preprocess the query using the pre_processing_text function
        processed_query = self.pre_processing_text(query)
        processed_tokens = re.findall(r'\b\w+\b', processed_query)
        
        # If there's no valid token after preprocessing, return an empty list
        if not processed_tokens:
            return []
        
        with InvertedIndexReader('main_index', self.postings_encoding, directory=self.output_dir) as index:
            # Total number of documents
            N = index.nilai_N
            
            # Dictionary to store document scores
            doc_scores = defaultdict(int)
            
            for token in processed_tokens:
                token = token.lower()
                # Check if the token exists in the term_id_map
                if token not in self.term_id_map:
                    continue
                
                # Get postings list and tf list for the term
                postings_list, tf_list = self.get_postings_list(token)

                # Calculate w(t, Q) using IDF formula
                df = len(postings_list)
                w_t_Q = math.log(N / df, 10)
                
                # For each document in postings list
                for doc_id, tf in zip(postings_list, tf_list):
                    # Calculate w(t, D)
                    w_t_D = (1 + math.log(tf, 10))
                    # Update document score
                    doc_scores[doc_id] += w_t_Q * w_t_D
                    
        # Sort the documents based on their scores in descending order
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top-K documents
        return [(score, self.doc_id_map[doc_id], doc_id) for doc_id, score in sorted_docs[:k]]




    def retrieve_bm25(self, query, k=10, k1=1.2, b=0.75):
        """
        Melakukan Ranked Retrieval dengan skema scoring BM25 dan framework TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        """
        with InvertedIndexReader('main_index', self.postings_encoding, directory=self.output_dir) as index:
            N = index.nilai_N
            # Preprocess the query using the pre_processing_text function
            processed_query = self.pre_processing_text(query)
            processed_tokens = re.findall(r'\b\w+\b', processed_query)
            
            # If there's no valid token after preprocessing, return an empty list
            if not processed_tokens:
                return []
            
            avgdl = sum(index.doc_length.values()) / N
            scores = defaultdict(int)

            
            for token in processed_tokens:
                token = token.lower()
                if token not in self.term_id_map:
                    continue
                
                postings_list, tf_list = self.get_postings_list(token)
                df = len(postings_list)
                idf = max(math.log((N - df + 0.5) / (df + 0.5), 10), 0)
                
                for doc_id, tf in zip(postings_list, tf_list):
                    doc_len = index.doc_length[doc_id]
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * doc_len / avgdl)
                    scores[doc_id] += idf * numerator / denominator

        # Sort documents based on scores in descending order
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top-K documents
        return [(score, self.doc_id_map[doc_id], doc_id) for doc_id, score in sorted_docs[:k]]

    def _get_upper_bound_tfidf(self, term):
        """Menghitung upper bound untuk TF-IDF dari term tertentu."""
        with InvertedIndexReader('main_index', self.postings_encoding, directory=self.output_dir) as index:
            N = index.nilai_N
            if term not in self.term_id_map:
                return 0
            postings_list, tf_list = self.get_postings_list(term)
            df = len(postings_list)
            idf = math.log(N / df, 10)
            max_tf = max(tf_list)
            return idf * (1 + math.log(max_tf, 10))
    
    def _get_upper_bound_bm25(self, term, k1, b):
        """Menghitung upper bound untuk BM25 dari term tertentu."""
        with InvertedIndexReader('main_index', self.postings_encoding, directory=self.output_dir) as index:
            N = index.nilai_N
            avgdl = sum(index.doc_length.values()) / N
            if term not in self.term_id_map:
                return 0
            postings_list, tf_list = self.get_postings_list(term)
            if not tf_list:
                return 0  # Return 0 if tf_list is empty
            df = len(postings_list)
            idf = max(math.log((N - df + 0.5) / (df + 0.5), 10), 0)
            max_tf = max(tf_list)
            score = idf * (max_tf * (k1 + 1)) / (max_tf + k1 * (1 - b + b * avgdl / avgdl))
            return score

    def retrieve_tfidf_wand(self, query, k=10):
        """Ranked Retrieval dengan TF-IDF menggunakan WAND Top-K Retrieval."""
        processed_query = self.pre_processing_text(query)
        processed_tokens = re.findall(r'\b\w+\b', processed_query)
        
        term_bounds = [(term, self._get_upper_bound_tfidf(term)) for term in processed_tokens]
        term_bounds = sorted(term_bounds, key=lambda x: x[1], reverse=True)  # Sort based on upper bounds
        
        # Initialization
        threshold = 0
        candidates = []
        
        for term, upper_bound in term_bounds:
            postings_list, tf_list = self.get_postings_list(term)
            for doc_id, tf in zip(postings_list, tf_list):
                score = upper_bound * (1 + math.log(tf, 10))
                if score > threshold:
                    heapq.heappush(candidates, (score, doc_id))
                    if len(candidates) > k:
                        threshold, _ = heapq.heappop(candidates)
        
        # Return top-K documents
        return [(score, self.doc_id_map[doc_id], doc_id) for score, doc_id in sorted(candidates, key=lambda x: x[0], reverse=True)]

    def retrieve_bm25_wand(self, query, k=10, k1=1.2, b=0.75):
        """Ranked Retrieval dengan BM25 menggunakan WAND Top-K Retrieval."""
        if len(query) <= 1:
            return []
        processed_query = self.pre_processing_text(query)
        processed_tokens = re.findall(r'\b\w+\b', processed_query)

        if len(processed_tokens) <= 0:
            return []

        self.load()

        term_bounds = []
        for term in processed_tokens:
            term_id = self.term_id_map[term]
            if term_id is not None:
                upper_bound = self._get_upper_bound_bm25(term, k1, b)
                term_bounds.append((term, upper_bound))

        term_bounds = sorted(term_bounds, key=lambda x: x[1], reverse=True)  # Sort based on upper bounds

        # Initialization
        threshold = 0
        candidates = []

        with InvertedIndexReader('main_index', self.postings_encoding, directory=self.output_dir) as index:
            avgdl = sum(index.doc_length.values()) / index.nilai_N

            for term, upper_bound in term_bounds:
                postings_list, tf_list = self.get_postings_list(term)
                if not tf_list:
                    continue  # Skip if tf_list is empty
                for doc_id, tf in zip(postings_list, tf_list):
                    doc_len = index.doc_length[doc_id]
                    score = upper_bound * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))
                    if score > threshold:
                        heapq.heappush(candidates, (score, doc_id))
                        if len(candidates) > k:
                            threshold, _ = heapq.heappop(candidates)

        # Return top-K documents
        return [(score, self.doc_id_map[doc_id], doc_id) for score, doc_id in sorted(candidates, key=lambda x: x[0], reverse=True)]


    def do_indexing(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parsing_block
        untuk parsing dokumen dan memanggil write_to_index yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parsing_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.output_dir) as index:
                self.write_to_index(td_pairs, index)
                td_pairs = None

        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                           for index_id in self.intermediate_indices]
                self.merge_index(indices, merged_index)
                merged_index.save_doc_lengths()


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir=os.path.join("retriever", "collections"),
                              postings_encoding=VBEPostings,
                              output_dir=os.path.join("retriever", "index"))
    BSBI_instance.save()
    BSBI_instance.do_indexing()  # memulai indexing!