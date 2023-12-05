import os
import pickle
import contextlib
import heapq
import time
import re
import sys

from .index import InvertedIndexReader, InvertedIndexWriter
from .util import IdMap, sort_intersect_list
from .compression import StandardPostings, VBEPostings

from mpstemmer import MPStemmer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

""" 
Ingat untuk install tqdm terlebih dahulu
pip intall tqdm
"""
from tqdm import tqdm

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_path(str): Path ke data
    output_path(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """
    def __init__(self, data_path, output_path, postings_encoding, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_path = data_path
        self.output_path = output_path
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""
        with open(os.path.join(self.output_path, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_path, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""
        with open(os.path.join(self.output_path, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_path, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def start_indexing(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_path in tqdm(sorted(next(os.walk(self.data_path))[1])):
            td_pairs = self.parsing_block(block_path)
            index_id = 'intermediate_index_'+block_path
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, path = self.output_path) as index:
                self.write_to_index(td_pairs, index)
                td_pairs = None
    
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, path = self.output_path) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, path=self.output_path))
                               for index_id in self.intermediate_indices]
                self.merge_index(indices, merged_index)

    def parsing_block(self, block_path):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Indonesia Seperti
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
        termIDs dan docIDs. Dua variable ini harus persis untuk semua pemanggilan
        parse_block(...).
        """
        # TODO
        stop_factory = StopWordRemoverFactory()
        stop_words_list: List[str] = stop_factory.get_stop_words()
        stop_words_set: Set[str] = set(stop_words_list)

        res: List[Tuple[int, int]] = []

        for cur_file in tqdm(os.listdir(self.data_path + block_path)):
            with open(os.path.join(self.data_path + block_path, cur_file), 'r') as f:
                file_path = os.path.join(self.data_path + block_path, cur_file)
                doc_id = self.doc_id_map[file_path]

                text = f.read()

                # tokenizing - uses regex for better performance (fast)
                tokenizer_pattern: str = r'\w+'
                tokens: List[str] = re.findall(tokenizer_pattern, text)

                # lemmatization and stemming using mpstemmer
                stemmer = MPStemmer()

                stemmed_tokens: List[str] = [stemmer.stem(token) if token else '' for token in tokens]
                stemmed_tokens: List[str] = [
                    token for token in stemmed_tokens
                    if not ((token == '') or (token == None))
                ]

                tokens_without_stop_words: List[str] = [token for token in stemmed_tokens if token not in stop_words_set]

                for token in tokens_without_stop_words:
                    term_id = self.term_id_map[token]
                    res.append((term_id, doc_id))

        return res

    def write_to_index(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        term_dict = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = set()
            term_dict[term_id].add(doc_id)
        for term_id in sorted(term_dict.keys()):
            index.append(term_id, sorted(list(term_dict[term_id])))

    def merge_index(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

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
        # TODO
        # buat heap (struktur data di memori) of 2-tuple: term_id and index_id
        # index id untuk indices[i] adalah i
        heap = []

        # insert 50% of each postings list in index into memory
        for i in range(len(indices)):
            indices[i].__iter__()
            for j in range(len(indices[i].terms) // 2):
                heapq.heappush(heap, (next(indices[i]), i))
        
        cur_term_postings_list = []
        cur_term_id = -1
        # masukin ke index baru
        # heapq.heappop balikannya 2-tuple
        # new_postings_list[0] itu term idnya
        # new_postings_list[1] itu list postingnya
        while len(heap) > 0:
            new_postings_list, index_id = heapq.heappop(heap)
            if new_postings_list[0] == cur_term_id:
                # kalau sama posting idnya gabungin 2 sorted list
                idx_cur = 0
                idx_new = 0
                merged_term_postings_list = []
                while idx_cur < len(cur_term_postings_list) and idx_new < len(new_postings_list[1]):
                    if (cur_term_postings_list[idx_cur] == new_postings_list[1][idx_new]):
                        merged_term_postings_list.append(cur_term_postings_list[idx_cur])
                        idx_cur += 1
                        idx_new += 1
                    elif (cur_term_postings_list[idx_cur] < new_postings_list[1][idx_new]):
                        merged_term_postings_list.append(cur_term_postings_list[idx_cur])
                        idx_cur += 1
                    else:
                        merged_term_postings_list.append(new_postings_list[1][idx_new])
                        idx_new += 1
                while idx_cur < len(cur_term_postings_list):
                    merged_term_postings_list.append(cur_term_postings_list[idx_cur])
                    idx_cur += 1
                while idx_new < len(new_postings_list[1]):
                    merged_term_postings_list.append(new_postings_list[1][idx_new])
                    idx_new += 1
                
                cur_term_postings_list = merged_term_postings_list

            else:
                # kalau beda saatnya masukin ke index baru
                merged_index.append(cur_term_id, cur_term_postings_list)
                cur_term_postings_list = new_postings_list[1]
                cur_term_id = new_postings_list[0]
            
            # masukkan item selanjutnya dari index yang baru saja dipop
            next_tuple = next(indices[index_id])
            if next_tuple[0] != None and next_tuple[0] != "":
                heapq.heappush(heap, (next_tuple, i))


    def boolean_retrieve(self, query):
        """
        Melakukan boolean retrieval untuk mengambil semua dokumen yang
        mengandung semua kata pada query. Jangan lupa lakukan pre-processing
        yang sama dengan yang dilakukan pada proses indexing!
        (Stemming dan Stopwords Removal)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya adalah
                    boolean query "universitas AND indonesia AND depok"

        Result
        ------
        List[str]
            Daftar dokumen terurut yang mengandung sebuah query tokens.
            Harus mengembalikan EMPTY LIST [] jika tidak ada yang match.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.
        """
        # TODO
        stop_factory = StopWordRemoverFactory()
        stop_words_list: List[str] = stop_factory.get_stop_words()
        stop_words_set: Set[str] = set(stop_words_list)


        # tokenizing - uses regex for better performance (fast)
        tokenizer_pattern: str = r'\w+'
        tokens: List[str] = re.findall(tokenizer_pattern, query)


        # lemmatization and stemming using mpstemmer
        stemmer = MPStemmer()

        stemmed_tokens: List[str] = [stemmer.stem(token) if token else '' for token in tokens]
        stemmed_tokens: List[str] = [
            token for token in stemmed_tokens
            if not ((token == '') or (token == None))
        ]

        tokens_without_stop_words: List[str] = [token for token in stemmed_tokens if token not in stop_words_set]

        final_query_list = tokens_without_stop_words
        print(final_query_list)

        self.load()

        query_ids = [self.term_id_map[q] for q in final_query_list]

        result_ids = []

        with InvertedIndexReader(self.index_name, self.postings_encoding, path = self.output_path) as index:
            postings_lists = [index.get_postings_list(q)[1] for q in query_ids]

            if len(postings_lists) == 0:
                return []
            result_ids = postings_lists[0]

            for i in range(1, len(postings_lists)):
                result_ids = sort_intersect_list(postings_lists[i], result_ids)

        result = [self.doc_id_map[d] for d in result_ids]

        return result, result_ids


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_path = 'retriever/collections/', \
                              postings_encoding = VBEPostings, \
                              output_path = 'retriever/index')
    BSBI_instance.start_indexing() # memulai indexing!
