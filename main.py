import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import numpy as np
import networkx as nx
import re
import os
import json


class MultiJournalSummarizer:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('indonesian'))
        self.stemmer = PorterStemmer()

        # Tambahan stopwords khusus untuk jurnal IT
        self.tech_stop_words = {
            'framework', 'implementation', 'system', 'algorithm', 'method',
            'data', 'result', 'analysis', 'propose', 'proposed', 'methodology',
            'abstract', 'conclusion', 'references', 'et', 'al', 'fig', 'table'
        }
        self.stop_words.update(self.tech_stop_words)

    def preprocess_journal(self, text):
        """Preprocessing untuk satu jurnal"""
        # Bersihkan teks
        text = self.clean_text(text)

        # Ekstrak bagian-bagian jurnal
        sections = self.extract_sections(text)

        # Tokenisasi dan normalisasi
        processed_sections = {}
        for section, content in sections.items():
            if content:
                sentences = sent_tokenize(content)
                processed_sentences = []
                for sentence in sentences:
                    # Preprocessing kalimat
                    words = word_tokenize(sentence.lower())
                    words = [self.stemmer.stem(word) for word in words
                             if word.isalnum() and word not in self.stop_words]
                    if words:
                        processed_sentences.append({
                            'original': sentence,
                            'processed': ' '.join(words),
                            'vector': None  # Akan diisi nanti
                        })
                processed_sections[section] = processed_sentences

        return processed_sections

    def clean_text(self, text):
        """Membersihkan teks dari elemen yang tidak diperlukan"""
        # Hapus referensi
        text = re.sub(r'\[\d+(,\s*\d+)*\]', '', text)
        text = re.sub(r'\(\d{4}\)', '', text)

        # Hapus referensi gambar dan tabel
        text = re.sub(r'(Figure|Fig\.|Table|Tabel)\s+\d+', '', text)

        # Hapus URL
        text = re.sub(r'http\S+|www.\S+', '', text)

        # Hapus karakter khusus tapi pertahankan tanda baca penting
        text = re.sub(r'[^\w\s\.,;?!]', ' ', text)

        # Normalisasi whitespace
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def extract_sections(self, text):
        """Mengekstrak bagian-bagian penting jurnal"""
        sections = {
            'abstract': '',
            'introduction': '',
            'methodology': '',
            'results': '',
            'discussion': '',
            'conclusion': ''
        }

        # Pattern untuk mendeteksi header section
        section_patterns = {
            'abstract': r'abstract|abstrak',
            'introduction': r'introduction|pendahuluan',
            'methodology': r'method|methodology|metode|metodologi',
            'results': r'results|hasil',
            'discussion': r'discussion|pembahasan|diskusi',
            'conclusion': r'conclusion|kesimpulan'
        }

        current_section = None
        lines = text.split('\n')

        for line in lines:
            line_lower = line.lower()
            # Deteksi section baru
            for section, pattern in section_patterns.items():
                if re.search(pattern, line_lower):
                    current_section = section
                    break

            # Tambahkan konten ke section yang sesuai
            if current_section and line:
                sections[current_section] += line + ' '

        return sections

    def calculate_similarity_matrix(self, sentences):
        """Menghitung similarity matrix antar kalimat"""
        # Gunakan TF-IDF untuk vectorization
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(
            [sent['processed'] for sent in sentences])
        similarity_matrix = (vectors * vectors.T).toarray()

        return similarity_matrix

    def rank_sentences(self, similarity_matrix, sentences, damping=0.85):
        """Meranking kalimat menggunakan TextRank"""
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph, alpha=damping)

        ranked_sentences = []
        for idx, score in scores.items():
            ranked_sentences.append({
                'sentence': sentences[idx]['original'],
                'score': score
            })

        return sorted(ranked_sentences, key=lambda x: x['score'], reverse=True)

    def get_section_weight(self, section_name):
        """Menentukan bobot untuk setiap bagian jurnal"""
        weights = {
            'abstract': 1.5,
            'introduction': 1.2,
            'methodology': 1.0,
            'results': 1.3,
            'discussion': 1.2,
            'conclusion': 1.4
        }
        return weights.get(section_name, 1.0)

    def summarize_multiple_journals(self, journal_texts, num_sentences_per_doc=5):
        """Membuat ringkasan dari multiple jurnal"""
        all_summaries = []
        common_themes = defaultdict(int)

        # Proses setiap jurnal
        for idx, journal_text in enumerate(journal_texts):
            # Preprocessing jurnal
            processed_sections = self.preprocess_journal(journal_text)

            # Kumpulkan semua kalimat dengan metadata
            all_sentences = []
            for section, sentences in processed_sections.items():
                section_weight = self.get_section_weight(section)
                for sent in sentences:
                    sent['section'] = section
                    sent['weight'] = section_weight
                    all_sentences.append(sent)

            # Hitung similarity dan ranking
            if all_sentences:
                similarity_matrix = self.calculate_similarity_matrix(
                    all_sentences)
                ranked_sentences = self.rank_sentences(
                    similarity_matrix, all_sentences)

                # Pilih kalimat terbaik dengan mempertimbangkan bobot section
                weighted_sentences = []
                for sent in ranked_sentences:
                    original_idx = next(i for i, s in enumerate(all_sentences)
                                        if s['original'] == sent['sentence'])
                    section_weight = all_sentences[original_idx]['weight']
                    weighted_sentences.append({
                        'sentence': sent['sentence'],
                        'score': sent['score'] * section_weight
                    })

                # Ambil top sentences
                top_sentences = sorted(weighted_sentences,
                                       key=lambda x: x['score'],
                                       reverse=True)[:num_sentences_per_doc]

                # Tambahkan ke ringkasan
                summary = {
                    'journal_id': idx + 1,
                    'summary': [sent['sentence'] for sent in top_sentences]
                }
                all_summaries.append(summary)

                # Identifikasi tema umum
                for sent in all_sentences:
                    words = sent['processed'].split()
                    for word in words:
                        common_themes[word] += 1

        # Identifikasi tema utama dari semua jurnal
        main_themes = sorted(common_themes.items(),
                             key=lambda x: x[1], reverse=True)[:10]

        return {
            'individual_summaries': all_summaries,
            'main_themes': main_themes
        }


def main():
    # Contoh penggunaan dengan multiple jurnal
    journal1 = """
    Abstract
    Penelitian ini mengusulkan metode deep learning baru untuk klasifikasi teks.
    
    Introduction
    Deep learning telah menunjukkan hasil yang menjanjikan dalam berbagai tugas NLP.
    
    Methodology
    Kami menggunakan arsitektur transformer dengan attention mechanism.
    
    Results
    Akurasi klasifikasi mencapai 92% pada dataset benchmark.
    
    Conclusion
    Metode yang diusulkan mengungguli baseline dengan margin signifikan.
    """

    journal2 = """
    Abstract
    Studi ini membahas implementasi blockchain dalam sistem keamanan data.
    
    Introduction
    Blockchain menawarkan solusi terdesentralisasi untuk masalah keamanan.
    
    Methodology
    Smart contract dikembangkan menggunakan Solidity.
    
    Results
    Sistem menunjukkan peningkatan keamanan sebesar 45%.
    
    Conclusion
    Blockchain terbukti efektif untuk mengamankan data sensitif.
    """

    # Inisialisasi summarizer
    summarizer = MultiJournalSummarizer()

    # Generate summary
    results = summarizer.summarize_multiple_journals([journal1, journal2])

    # Tampilkan hasil
    print("=== Ringkasan Multiple Jurnal ===")
    for summary in results['individual_summaries']:
        print(f"\nJurnal {summary['journal_id']}:")
        for sentence in summary['summary']:
            print(f"- {sentence}")

    print("\nTema Utama:")
    for theme, count in results['main_themes']:
        print(f"- {theme}: {count} kemunculan")


if __name__ == "__main__":
    main()
