import array

class StandardPostings:
    """ 
    Class dengan static methods, untuk mengubah representasi postings list
    yang awalnya adalah List of integer, berubah menjadi sequence of bytes.
    Kita menggunakan Library array di Python.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    Silakan pelajari:
        https://docs.python.org/3/library/array.html
    """

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        # Untuk yang standard, gunakan L untuk unsigned long, karena docID
        # tidak akan negatif. Dan kita asumsikan docID yang paling besar
        # cukup ditampung di representasi 4 byte unsigned.
        return array.array('L', postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = array.array('L')
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()


class VBEPostings:
    """ 
    Berbeda dengan StandardPostings, dimana untuk suatu postings list,
    yang disimpan di disk adalah sequence of integers asli dari postings
    list tersebut apa adanya.

    Pada VBEPostings, kali ini, yang disimpan adalah gap-nya, kecuali
    posting yang pertama. Barulah setelah itu di-encode dengan Variable-Byte
    Enconding algorithm ke bytestream.

    Contoh:
    postings list [34, 67, 89, 454] akan diubah dulu menjadi gap-based,
    yaitu [34, 33, 22, 365]. Barulah setelah itu di-encode dengan algoritma
    compression Variable-Byte Encoding, dan kemudian diubah ke bytesream.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    """

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes (dengan Variable-Byte
        Encoding). JANGAN LUPA diubah dulu ke gap-based list, sebelum
        di-encode dan diubah ke bytearray.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        # TODO
        postings_list_gaps: List[int] = []

        if len(postings_list) > 0:
            postings_list_gaps.append(postings_list[0])

        for i in range(1, len(postings_list)):
            postings_list_gaps.append(postings_list[i] - postings_list[i-1])

        return VBEPostings.vb_encode(postings_list_gaps)
    
    @staticmethod
    def vb_encode(list_of_numbers):
        """ 
        Melakukan encoding (tentunya dengan compression) terhadap
        list of numbers, dengan Variable-Byte Encoding
        """
        # TODO
        encoded_postings_list = bytearray()
        for number in list_of_numbers:
            encoded_postings_list.extend(VBEPostings.vb_encode_number(number))

        return bytes(encoded_postings_list)

    @staticmethod
    def vb_encode_number(number):
        """
        Encodes a number using Variable-Byte Encoding
        Lihat buku teks kita!
        """
        # TODO
        res = bytearray()
        res.append(128 + (number % 128))
        while (number >= 128):
            number //= 128
            res.append(number % 128)

        res.reverse()

        return res

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes. JANGAN LUPA
        bytestream yang di-decode dari encoded_postings_list masih berupa
        gap-based list.

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        # TODO
        decoded_postings_list = VBEPostings.vb_decode(encoded_postings_list)
        postings_list_with_no_gap: List[int] = []

        if len(decoded_postings_list) > 0:
            postings_list_with_no_gap.append(decoded_postings_list[0])

        for i in range(1, len(decoded_postings_list)):
            postings_list_with_no_gap.append(decoded_postings_list[i] + postings_list_with_no_gap[i-1])

        return postings_list_with_no_gap

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Decoding sebuah bytestream yang sebelumnya di-encode dengan
        variable-byte encoding.
        """
        numbers = []
        n = 0
        for byte in encoded_bytestream:
            if (byte < 128):
                n = 128 * n + byte
            else:
                n = 128 * n + byte - 128
                numbers.append(n)
                n = 0
        return numbers

    

if __name__ == '__main__':
    
    postings_list = [34, 67, 89, 454, 2345738]
    for Postings in [StandardPostings, VBEPostings]:
        print(Postings.__name__)
        encoded_postings_list = Postings.encode(postings_list)
        print("byte hasil encode: ", encoded_postings_list)
        print("ukuran encoded postings: ", len(encoded_postings_list), "bytes")
        decoded_posting_list = Postings.decode(encoded_postings_list)
        print("hasil decoding: ", decoded_posting_list)
        assert decoded_posting_list == postings_list, "hasil decoding tidak sama dengan postings original"
        print()
