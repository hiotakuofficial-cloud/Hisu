"""
Dataset Downloader Module
Handles downloading and caching of anime and Hindi language datasets
"""

import os
import requests
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
import hashlib
import json


class DatasetDownloader:
    """Downloads and manages datasets for anime and Hindi language training"""

    def __init__(self, cache_dir: str = "data/raw"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "dataset_metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load dataset metadata from cache"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save dataset metadata to cache"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def download_file(self, url: str, filename: str, force: bool = False) -> Path:
        """Download file from URL with caching"""
        filepath = self.cache_dir / filename

        # Check if already downloaded
        if filepath.exists() and not force:
            print(f"✓ {filename} already cached")
            return filepath

        print(f"Downloading {filename}...")
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"✓ Downloaded {filename}")
            return filepath
        except Exception as e:
            print(f"✗ Error downloading {filename}: {str(e)}")
            return None

    def create_sample_anime_dataset(self) -> pd.DataFrame:
        """Create a sample anime dataset for demonstration"""
        data = {
            'anime_id': range(1, 101),
            'title': [f'Anime Title {i}' for i in range(1, 101)],
            'title_english': [f'English Title {i}' for i in range(1, 101)],
            'title_hindi': [f'हिंदी शीर्षक {i}' for i in range(1, 101)],
            'genre': ['Action, Adventure', 'Drama, Romance', 'Comedy, Slice of Life', 'Fantasy, Magic'] * 25,
            'type': ['TV', 'Movie', 'OVA', 'Special'] * 25,
            'episodes': [12, 24, 1, 48] * 25,
            'rating': [7.5, 8.2, 6.8, 9.1] * 25,
            'members': [100000, 250000, 50000, 500000] * 25,
            'synopsis': [f'This is a synopsis for anime {i}. It follows the story of characters in an exciting world.' for i in range(1, 101)],
            'synopsis_hindi': [f'यह एनीमे {i} का सारांश है। यह एक रोमांचक दुनिया में पात्रों की कहानी का अनुसरण करता है।' for i in range(1, 101)],
            'aired': ['2020-01-01', '2021-06-15', '2019-04-10', '2022-10-05'] * 25,
            'studios': ['Studio A', 'Studio B', 'Studio C', 'Studio D'] * 25,
        }
        return pd.DataFrame(data)

    def create_sample_hindi_translation_dataset(self) -> pd.DataFrame:
        """Create sample Hindi-English parallel corpus for NLP training"""
        data = {
            'english': [
                'What is your favorite anime?',
                'I love watching action anime.',
                'The animation quality is excellent.',
                'This character is very interesting.',
                'The story is very engaging.',
                'I recommend this series to everyone.',
                'The soundtrack is amazing.',
                'This is a popular anime series.',
                'Have you seen this episode?',
                'The plot twist was unexpected.',
                'I am currently watching a new anime.',
                'This anime has great character development.',
                'The art style is unique.',
                'I enjoy fantasy anime the most.',
                'This is my favorite character.',
                'The voice acting is superb.',
                'I can\'t wait for the next season.',
                'This anime is highly rated.',
                'The manga is even better.',
                'I have watched all episodes.',
            ] * 5,
            'hindi': [
                'आपका पसंदीदा एनीमे क्या है?',
                'मुझे एक्शन एनीमे देखना बहुत पसंद है।',
                'एनीमेशन की गुणवत्ता उत्कृष्ट है।',
                'यह चरित्र बहुत दिलचस्प है।',
                'कहानी बहुत आकर्षक है।',
                'मैं इस श्रृंखला की सिफारिश सभी को करता हूं।',
                'साउंडट्रैक अद्भुत है।',
                'यह एक लोकप्रिय एनीमे श्रृंखला है।',
                'क्या आपने यह एपिसोड देखा है?',
                'कथानक का मोड़ अप्रत्याशित था।',
                'मैं वर्तमान में एक नई एनीमे देख रहा हूं।',
                'इस एनीमे में चरित्र विकास बहुत अच्छा है।',
                'कला शैली अनूठी है।',
                'मुझे फंतासी एनीमे सबसे ज्यादा पसंद है।',
                'यह मेरा पसंदीदा चरित्र है।',
                'आवाज अभिनय शानदार है।',
                'मैं अगले सीज़न का इंतजार नहीं कर सकता।',
                'यह एनीमे बहुत उच्च रेटेड है।',
                'मांगा और भी बेहतर है।',
                'मैंने सभी एपिसोड देख लिए हैं।',
            ] * 5,
            'context': ['conversation'] * 100
        }
        return pd.DataFrame(data)

    def create_comprehensive_training_dataset(self) -> pd.DataFrame:
        """Create comprehensive dataset combining anime info with Hindi translations"""
        anime_df = self.create_sample_anime_dataset()
        hindi_df = self.create_sample_hindi_translation_dataset()

        # Save datasets
        anime_path = self.cache_dir / "anime_dataset.csv"
        hindi_path = self.cache_dir / "hindi_translation_dataset.csv"

        anime_df.to_csv(anime_path, index=False, encoding='utf-8')
        hindi_df.to_csv(hindi_path, index=False, encoding='utf-8')

        print(f"✓ Saved anime dataset: {anime_path} ({len(anime_df)} records)")
        print(f"✓ Saved Hindi translation dataset: {hindi_path} ({len(hindi_df)} records)")

        # Update metadata
        self.metadata['anime_dataset'] = {
            'path': str(anime_path),
            'records': len(anime_df),
            'columns': list(anime_df.columns)
        }
        self.metadata['hindi_dataset'] = {
            'path': str(hindi_path),
            'records': len(hindi_df),
            'columns': list(hindi_df.columns)
        }
        self._save_metadata()

        return anime_df, hindi_df

    def get_dataset_info(self) -> Dict:
        """Get information about downloaded datasets"""
        return self.metadata

    def load_anime_dataset(self) -> Optional[pd.DataFrame]:
        """Load anime dataset from cache"""
        if 'anime_dataset' in self.metadata:
            path = Path(self.metadata['anime_dataset']['path'])
            if path.exists():
                return pd.read_csv(path, encoding='utf-8')
        return None

    def load_hindi_dataset(self) -> Optional[pd.DataFrame]:
        """Load Hindi translation dataset from cache"""
        if 'hindi_dataset' in self.metadata:
            path = Path(self.metadata['hindi_dataset']['path'])
            if path.exists():
                return pd.read_csv(path, encoding='utf-8')
        return None


# Instructions for using real datasets from Kaggle/HuggingFace:
"""
DATASET SOURCES AVAILABLE:

1. ANIME DATASETS (Kaggle):
   - Anime Dataset 2023: kaggle.com/datasets/dbdmobile/myanimelist-dataset
   - MyAnimeList Dataset: kaggle.com/datasets/svanoo/myanimelist-dataset
   - Manga & Anime 2024: kaggle.com/datasets/duongtruongbinh/manga-and-anime-dataset
   - Anime Recommendations Database: kaggle.com/datasets/CooperUnion/anime-recommendations-database

   Download using Kaggle API:
   ```bash
   pip install kaggle
   kaggle datasets download -d dbdmobile/myanimelist-dataset
   ```

2. HINDI TRANSLATION DATASETS:
   - IIT Bombay English-Hindi Parallel Corpus:
     URL: https://www.cfilt.iitb.ac.in/iitb_parallel/
     HuggingFace: huggingface.co/datasets/cfilt/iitb-english-hindi

   - AI4Bharat Samanantar (46M parallel sentences):
     URL: https://ai4bharat.iitm.ac.in/samanantar

   Download using HuggingFace:
   ```python
   from datasets import load_dataset
   dataset = load_dataset("cfilt/iitb-english-hindi")
   ```

3. USAGE:
   Place downloaded CSV files in data/raw/ directory
   Update paths in config to point to real datasets
"""
