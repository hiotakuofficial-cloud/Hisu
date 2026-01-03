"""
Expand datasets to reach 5000+ high-quality training examples
"""

import csv
import random
from pathlib import Path


class DatasetExpander:
    """Expands existing datasets with more diverse examples"""
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
    
    def expand_all_datasets(self):
        """Expand all datasets to reach target sizes"""
        print("=" * 80)
        print("EXPANDING DATASETS TO 5000+ EXAMPLES")
        print("=" * 80)
        
        self.expand_conversational_dataset()
        self.expand_anime_dataset()
        self.expand_hindi_english_dataset()
        self.expand_anti_hallucination_dataset()
        
        # Count total
        total = self.count_total_examples()
        print("\n" + "=" * 80)
        print(f"TOTAL DATASET SIZE: {total} examples")
        print("=" * 80)
    
    def expand_conversational_dataset(self):
        """Expand conversational dataset to 2000+ examples"""
        print("\n[1/4] Expanding Conversational Dataset...")
        
        file_path = self.output_dir / "conversational_dataset.csv"
        
        # Read existing data
        existing_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_data = list(reader)
        
        new_data = []
        
        # Add more varied greetings
        greetings_variations = [
            ("Hi", "Hi! How can I assist you today?"),
            ("Hey there", "Hey! What can I help you with?"),
            ("Good day", "Good day! How may I help you?"),
            ("Yo", "Hello! What's on your mind?"),
            ("Sup", "Hey! What can I do for you?"),
            ("Hiya", "Hi there! How can I help?"),
            ("Hello friend", "Hello! I'm here to help you."),
            ("Greetings friend", "Greetings! What brings you here?"),
            ("Hi bot", "Hi! I'm ready to assist you."),
            ("Hello AI", "Hello! How can I help you today?"),
        ]
        new_data.extend([{'input': i, 'output': o, 'category': 'general'} for i, o in greetings_variations])
        
        # Add capability questions
        capability_questions = [
            ("What are your capabilities?", "I can help with anime recommendations, answer questions about anime series, discuss genres and characters, and converse in both English and Hindi."),
            ("What do you know?", "I have knowledge about anime series, genres, characters, studios, and can provide recommendations based on your preferences."),
            ("How can you assist me?", "I can recommend anime, answer questions about specific series, explain anime concepts, and have conversations in English or Hindi."),
            ("What services do you provide?", "I provide anime recommendations, information about anime series and characters, genre explanations, and multilingual conversation support."),
            ("Are you helpful?", "I strive to be helpful by providing accurate anime recommendations and information based on your interests."),
            ("Can you recommend anime?", "Yes, I can recommend anime based on your preferences. What genre or type of anime interests you?"),
            ("Do you know about anime?", "Yes, I have extensive knowledge about anime series, genres, characters, studios, and industry information."),
            ("Can you speak multiple languages?", "Yes, I can communicate in both English and Hindi, and understand code-switching between them."),
            ("What makes you useful?", "I can help you discover new anime, learn about series you're interested in, and have engaging conversations about anime culture."),
            ("Why should I use you?", "I can save you time finding anime that match your preferences and provide detailed information about series you're curious about."),
        ]
        new_data.extend([{'input': i, 'output': o, 'category': 'general'} for i, o in capability_questions])
        
        # Add preference-based questions
        preference_questions = [
            ("I like dark themes", "Dark-themed anime can be very compelling. Would you like recommendations for psychological thrillers, horror, or dark fantasy?"),
            ("I prefer lighthearted shows", "Lighthearted anime are great for relaxation! Would you like comedy, slice of life, or feel-good recommendations?"),
            ("I enjoy complex plots", "Complex narratives can be very engaging. Would you like mystery, psychological, or multi-layered story recommendations?"),
            ("I like simple stories", "Simple, straightforward stories can be very enjoyable. Would you like slice of life, comedy, or adventure recommendations?"),
            ("I want character development", "Character-driven stories are excellent. Would you like drama, coming-of-age, or character study recommendations?"),
            ("I prefer action over story", "Action-focused anime can be thrilling! Would you like battle shounen, martial arts, or pure action recommendations?"),
            ("I like beautiful animation", "Visual quality matters! Would you like recommendations from studios known for exceptional animation like Ufotable or KyoAni?"),
            ("I don't mind old anime", "Classic anime have timeless appeal! Would you like recommendations from the 90s, 2000s, or earlier?"),
            ("I only watch new anime", "Recent anime have great production values! Would you like recommendations from the last few years?"),
            ("I like unique art styles", "Distinctive art styles can enhance storytelling! Would you like recommendations for visually unique anime?"),
        ]
        new_data.extend([{'input': i, 'output': o, 'category': 'general'} for i, o in preference_questions])
        
        # Add follow-up questions
        followup_questions = [
            ("Tell me more", "I'd be happy to provide more details. What specific aspect would you like to know more about?"),
            ("Can you elaborate?", "Certainly! Which part would you like me to explain in more detail?"),
            ("What else?", "There's more to discuss! What other information would you like?"),
            ("Continue", "I'll continue. What would you like me to focus on?"),
            ("Go on", "Sure! What aspect interests you most?"),
            ("Interesting", "I'm glad you find it interesting! Would you like to know more?"),
            ("That sounds good", "Great! Would you like me to provide more recommendations or information?"),
            ("I see", "Do you have any questions or would you like additional information?"),
            ("Okay", "Is there anything else I can help you with?"),
            ("Got it", "Perfect! What else would you like to know?"),
        ]
        new_data.extend([{'input': i, 'output': o, 'category': 'general'} for i, o in followup_questions])
        
        # Add emotional states
        emotional_states = [
            ("I'm sad", "I'm sorry you're feeling sad. Would you like me to recommend some uplifting or comforting anime?"),
            ("I'm happy", "That's wonderful! Would you like recommendations for fun, energetic anime to match your mood?"),
            ("I'm stressed", "I understand. Would you like relaxing slice of life anime or something to take your mind off things?"),
            ("I'm excited", "Great energy! Would you like exciting action or adventure anime recommendations?"),
            ("I'm tired", "When you're tired, easy-to-watch anime can be perfect. Would you like lighthearted or calming recommendations?"),
            ("I'm curious", "Curiosity is great! What aspect of anime would you like to explore?"),
            ("I'm confused", "No problem! What can I clarify for you?"),
            ("I'm frustrated", "I understand. How can I help make things easier for you?"),
            ("I'm bored", "Let's fix that! What type of entertainment are you in the mood for?"),
            ("I'm nostalgic", "Nostalgia can be wonderful. Would you like recommendations for classic anime or series that evoke nostalgic feelings?"),
        ]
        new_data.extend([{'input': i, 'output': o, 'category': 'general'} for i, o in emotional_states])
        
        # Add time-based queries
        time_queries = [
            ("I have 30 minutes", "Perfect! I can recommend anime movies or a few episodes of a series. What genre interests you?"),
            ("I have all day", "Great! You could start a new series or watch multiple movies. What are you in the mood for?"),
            ("I want something quick", "For quick entertainment, I can recommend short anime series (12 episodes) or movies. What genre?"),
            ("I want a long series", "Long series offer deep storytelling! Would you like shounen, adventure, or other long-running recommendations?"),
            ("What can I watch in an hour?", "In an hour, you can watch an anime movie or 2-3 episodes of a series. What genre interests you?"),
            ("I have a weekend free", "A free weekend is perfect for binge-watching! Would you like a complete short series or starting a longer one?"),
            ("Quick recommendation", "For a quick recommendation, what genre do you prefer?"),
            ("I need something now", "I can give you an immediate recommendation! What type of anime do you enjoy?"),
            ("What's a good movie?", "There are many great anime movies! What genre or theme interests you?"),
            ("Suggest a short series", "Short series (12-13 episodes) are perfect for quick completion. What genre would you like?"),
        ]
        new_data.extend([{'input': i, 'output': o, 'category': 'general'} for i, o in time_queries])
        
        # Add experience level questions
        experience_questions = [
            ("I'm an anime expert", "Great! Are you looking for hidden gems, recent releases, or specific niche genres?"),
            ("I've seen a lot of anime", "Excellent! Would you like recommendations for lesser-known series or specific genres you haven't explored?"),
            ("I've only seen a few anime", "That's a great start! Would you like recommendations to expand your anime experience?"),
            ("I've never watched anime", "Welcome to anime! I can recommend beginner-friendly series. What genres do you usually enjoy?"),
            ("I'm a casual viewer", "Perfect! Would you like accessible, easy-to-follow anime recommendations?"),
            ("I'm a hardcore fan", "Awesome! Are you looking for deep cuts, specific genres, or challenging narratives?"),
            ("I watch occasionally", "That's fine! Would you like recommendations for standalone series or movies?"),
            ("I'm getting back into anime", "Welcome back! Would you like to catch up on recent popular series or revisit classic genres?"),
            ("I used to watch as a kid", "Nostalgia! Would you like recommendations similar to what you watched, or explore new genres?"),
            ("I'm exploring anime", "Exploration is exciting! What aspects of anime interest you most?"),
        ]
        new_data.extend([{'input': i, 'output': o, 'category': 'general'} for i, o in experience_questions])
        
        # Add comparison questions
        comparison_questions = [
            ("What's similar to Naruto?", "Similar to Naruto, you might enjoy My Hero Academia, Black Clover, or Demon Slayer - all feature young protagonists growing stronger."),
            ("Like Death Note but different", "If you liked Death Note's psychological elements, try Code Geass, Monster, or Psycho-Pass for similar strategic thinking."),
            ("Something like Attack on Titan", "For Attack on Titan's intensity, try Vinland Saga, Kabaneri of the Iron Fortress, or Tokyo Ghoul."),
            ("Similar to One Piece", "For One Piece's adventure spirit, try Fairy Tail, Hunter x Hunter, or Magi: The Labyrinth of Magic."),
            ("Like Fullmetal Alchemist", "For FMA's depth, try Steins;Gate, Made in Abyss, or The Promised Neverland."),
            ("Something like Your Name", "For Your Name's romance and beauty, try Weathering with You, A Silent Voice, or 5 Centimeters per Second."),
            ("Similar to Demon Slayer", "For Demon Slayer's action and animation, try Jujutsu Kaisen, Kimetsu no Yaiba, or God of High School."),
            ("Like My Hero Academia", "For MHA's superhero theme, try One Punch Man, Mob Psycho 100, or Tiger & Bunny."),
            ("Something like Steins;Gate", "For Steins;Gate's sci-fi mystery, try Re:Zero, Erased, or The Girl Who Leapt Through Time."),
            ("Similar to Cowboy Bebop", "For Cowboy Bebop's style, try Samurai Champloo, Space Dandy, or Trigun."),
        ]
        new_data.extend([{'input': i, 'output': o, 'category': 'general'} for i, o in comparison_questions])
        
        # Add opinion questions
        opinion_questions = [
            ("What do you think of Naruto?", "Naruto is widely regarded as an influential shounen anime with strong themes of perseverance and friendship, though opinions on pacing vary."),
            ("Is Attack on Titan good?", "Attack on Titan is highly acclaimed for its intense plot, complex themes, and character development, though it can be dark and violent."),
            ("Is One Piece worth it?", "One Piece is beloved for its world-building and character development, though its length can be daunting for some viewers."),
            ("What about Death Note?", "Death Note is praised for its psychological depth and strategic battles, considered a must-watch thriller anime."),
            ("Is My Hero Academia overrated?", "My Hero Academia is popular for good reasons - solid characters and action - though some find it formulaic. Enjoyment is subjective."),
            ("Is anime art?", "Anime is definitely an art form, combining visual artistry, storytelling, music, and voice acting to create unique experiences."),
            ("Why is anime popular?", "Anime's popularity stems from diverse storytelling, unique art styles, emotional depth, and ability to explore themes other media might avoid."),
            ("Is sub better than dub?", "Both have merits - subs preserve original performances, dubs offer accessibility. It's personal preference."),
            ("Are anime fans weird?", "Anime fans are diverse people who enjoy a particular medium, just like fans of any entertainment form."),
            ("Is watching anime childish?", "Anime spans all demographics with content for children, teens, and adults. It's a medium, not a genre limited to any age."),
        ]
        new_data.extend([{'input': i, 'output': o, 'category': 'general'} for i, o in opinion_questions])
        
        # Combine with existing data
        all_data = existing_data + new_data
        
        # Write back
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['input', 'output', 'category'])
            writer.writeheader()
            writer.writerows(all_data)
        
        print(f"✓ Expanded to {len(all_data)} examples (added {len(new_data)})")
    
    def expand_anime_dataset(self):
        """Expand anime dataset to 1500+ examples"""
        print("\n[2/4] Expanding Anime Dataset...")
        
        file_path = self.output_dir / "anime_dataset.csv"
        
        # Read existing
        existing_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_data = list(reader)
        
        new_data = []
        
        # Add more anime information
        more_anime_info = [
            ("Tell me about Hunter x Hunter", "Hunter x Hunter follows Gon's journey to become a Hunter and find his father. It's known for complex power systems and strategic battles."),
            ("What is Bleach about?", "Bleach follows Ichigo, who gains Soul Reaper powers and protects the living world from evil spirits while navigating the afterlife."),
            ("Describe Fairy Tail", "Fairy Tail is about a magical guild and its members' adventures, featuring friendship themes and magical battles."),
            ("What is Code Geass?", "Code Geass follows Lelouch, who gains the power to command anyone and leads a rebellion against an empire using strategy and manipulation."),
            ("Tell me about Tokyo Ghoul", "Tokyo Ghoul follows Kaneki, who becomes half-ghoul and must navigate between human and ghoul worlds while struggling with his identity."),
            ("What is Mob Psycho 100?", "Mob Psycho 100 follows a powerful psychic trying to live normally while dealing with his emotions and supernatural threats."),
            ("Describe Jujutsu Kaisen", "Jujutsu Kaisen follows Yuji, who becomes a vessel for a powerful curse and joins sorcerers fighting cursed spirits."),
            ("What is Vinland Saga about?", "Vinland Saga is a historical anime about Vikings, following Thorfinn's journey from revenge to finding purpose and peace."),
            ("Tell me about Re:Zero", "Re:Zero follows Subaru, who is transported to a fantasy world and gains the ability to return from death, but must relive trauma repeatedly."),
            ("What is The Promised Neverland?", "The Promised Neverland follows orphans who discover their orphanage's dark secret and plan an escape while outsmarting their caretakers."),
        ]
        new_data.extend([{'input': i, 'output': o, 'category': 'information'} for i, o in more_anime_info])
        
        # Add genre-specific recommendations
        genre_recommendations = [
            ("Recommend a thriller anime", "I recommend Monster - a psychological thriller about a doctor hunting a serial killer he once saved, with complex moral themes."),
            ("Suggest a sci-fi anime", "I suggest Steins;Gate - a brilliant sci-fi thriller about time travel with excellent character development and plot twists."),
            ("What's a good drama anime?", "Clannad: After Story is an emotional drama about family, love, and life's challenges with powerful storytelling."),
            ("Recommend a supernatural anime", "I recommend Mob Psycho 100 - a supernatural action-comedy with amazing animation and heartfelt character growth."),
            ("Suggest a historical anime", "Vinland Saga is an excellent historical anime set in Viking age with mature themes and character development."),
            ("What's a good adventure anime?", "Made in Abyss is a dark adventure anime with beautiful world-building and emotional depth despite its cute art style."),
            ("Recommend a school anime", "Assassination Classroom is a unique school anime about students trying to assassinate their alien teacher before he destroys Earth."),
            ("Suggest a music anime", "Your Lie in April is a beautiful music anime about a pianist overcoming trauma through music and relationships."),
            ("What's a good martial arts anime?", "Kengan Ashura is an intense martial arts anime featuring underground fighting tournaments with diverse fighting styles."),
            ("Recommend a cooking anime", "Food Wars! (Shokugeki no Soma) is an exciting cooking anime with over-the-top food reactions and culinary battles."),
        ]
        new_data.extend([{'input': i, 'output': o, 'category': 'recommendation'} for i, o in genre_recommendations])
        
        # Add character analysis
        character_analysis = [
            ("Why is Goku popular?", "Goku is popular for his pure-hearted nature, determination to grow stronger, and iconic status as a foundational shounen protagonist."),
            ("What makes Lelouch interesting?", "Lelouch is compelling due to his intelligence, moral complexity, strategic mind, and the consequences of his choices."),
            ("Why do people like Saitama?", "Saitama is beloved for subverting superhero tropes, his deadpan humor, and the series' satire of power scaling."),
            ("What's special about Mob?", "Mob is special for his emotional journey, kindness despite power, and the series' message about self-improvement beyond strength."),
            ("Why is L memorable?", "L is memorable for his eccentric behavior, brilliant deduction skills, and compelling rivalry with Light Yagami."),
            ("What makes Levi popular?", "Levi is popular for his combat prowess, cool demeanor, tragic backstory, and moments of unexpected depth."),
            ("Why do fans love Kakashi?", "Kakashi is loved for his mysterious past, cool abilities, mentorship role, and balance of humor and seriousness."),
            ("What's interesting about Senku?", "Senku is interesting for his scientific approach to problems, enthusiasm for knowledge, and unique protagonist archetype."),
            ("Why is Itachi complex?", "Itachi is complex due to his hidden motivations, sacrifice for the greater good, and the moral ambiguity of his actions."),
            ("What makes All Might inspiring?", "All Might inspires through his Symbol of Peace role, mentorship, and maintaining hope despite personal struggles."),
        ]
        new_data.extend([{'input': i, 'output': o, 'category': 'character'} for i, o in character_analysis])
        
        # Add theme discussions
        theme_discussions = [
            ("What anime explore friendship?", "Naruto, One Piece, and Fairy Tail strongly emphasize friendship as a core theme throughout their narratives."),
            ("Anime about redemption?", "Vinland Saga, Rurouni Kenshin, and Code Geass explore redemption through characters seeking to atone for past actions."),
            ("What anime deal with identity?", "Tokyo Ghoul, Mob Psycho 100, and Neon Genesis Evangelion deeply explore identity and self-acceptance themes."),
            ("Anime about growing up?", "March Comes in Like a Lion, A Silent Voice, and Fruits Basket beautifully portray coming-of-age and personal growth."),
            ("What anime explore morality?", "Death Note, Psycho-Pass, and Fate/Zero present complex moral dilemmas and ethical questions."),
            ("Anime about family?", "Clannad, Fullmetal Alchemist, and Spy x Family explore various aspects of family bonds and relationships."),
            ("What anime deal with loss?", "Anohana, Your Lie in April, and Violet Evergarden powerfully explore grief and coping with loss."),
            ("Anime about dreams?", "Bakuman, Shirobako, and Haikyuu!! follow characters pursuing their dreams with determination."),
            ("What anime explore war?", "Attack on Titan, Fullmetal Alchemist, and Gundam series examine war's impact and moral complexity."),
            ("Anime about loneliness?", "Welcome to the NHK, March Comes in Like a Lion, and Neon Genesis Evangelion explore isolation and connection."),
        ]
        new_data.extend([{'input': i, 'output': o, 'category': 'genre'} for i, o in theme_discussions])
        
        # Combine and write
        all_data = existing_data + new_data
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['input', 'output', 'category'])
            writer.writeheader()
            writer.writerows(all_data)
        
        print(f"✓ Expanded to {len(all_data)} examples (added {len(new_data)})")
    
    def expand_hindi_english_dataset(self):
        """Expand Hindi-English dataset to 1000+ examples"""
        print("\n[3/4] Expanding Hindi-English Dataset...")
        
        file_path = self.output_dir / "hindi_english_dataset.csv"
        
        # Read existing
        existing_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_data = list(reader)
        
        new_data = []
        
        # Add more Hindi conversations
        more_hindi = [
            ("आप कैसे काम करते हैं?", "मैं एक AI मॉडल हूं जो एनीमे और बातचीत के डेटा पर प्रशिक्षित है। मैं आपके सवालों को समझता हूं और उपयोगी जवाब देता हूं।"),
            ("मुझे कुछ नया चाहिए", "बिल्कुल! आप किस तरह की नई एनीमे देखना चाहेंगे? एक्शन, रोमांस, या कुछ और?"),
            ("यह बहुत अच्छा है", "धन्यवाद! क्या मैं और कुछ मदद कर सकता हूं?"),
            ("मैं समझ गया", "बढ़िया! क्या आपके कोई और सवाल हैं?"),
            ("यह दिलचस्प है", "मुझे खुशी है कि आपको दिलचस्प लगा! और जानना चाहेंगे?"),
            ("मुझे यह पसंद है", "बहुत अच्छा! इसी तरह की और एनीमे चाहिए?"),
            ("यह मुश्किल है", "मैं समझता हूं। मैं इसे आसान तरीके से समझाने की कोशिश करता हूं।"),
            ("मुझे मज़ा आया", "यह सुनकर अच्छा लगा! क्या आप कुछ और देखना चाहेंगे?"),
            ("यह बोरिंग है", "मैं समझता हूं। चलिए कुछ और रोमांचक ढूंढते हैं।"),
            ("मुझे डर लगता है", "कोई बात नहीं। क्या आप कम डरावनी एनीमे पसंद करेंगे?"),
        ]
        new_data.extend([{'input': i, 'output': o, 'category': 'hindi_greeting'} for i, o in more_hindi])
        
        # Add more code-switching
        more_code_switching = [
            ("मुझे thriller anime पसंद है", "Great choice! Thriller anime बहुत exciting होते हैं। मैं Death Note, Monster, या Steins;Gate recommend करता हूं।"),
            ("Slice of life क्या होता है?", "Slice of life anime रोजमर्रा की जिंदगी को दिखाते हैं। ये calm और relatable होते हैं, जैसे Barakamon या Non Non Biyori।"),
            ("मुझे sad ending पसंद नहीं", "मैं समझता हूं। मैं happy ending वाली anime suggest करूंगा जैसे Haikyuu!! या My Love Story!!"),
            ("Popular anime कौन से हैं?", "Currently popular anime में Jujutsu Kaisen, Demon Slayer, और Attack on Titan हैं।"),
            ("मैं binge watch करना चाहता हूं", "Perfect! मैं complete series recommend करता हूं जो binge-worthy हैं।"),
            ("Short anime बताओ", "Short anime के लिए Erased (12 episodes), Death Parade (12 episodes), या Devilman Crybaby (10 episodes) try करें।"),
            ("मुझे emotional anime चाहिए", "Emotional anime के लिए Your Lie in April, Clannad, या Anohana perfect हैं।"),
            ("Action packed कुछ बताओ", "Action-packed anime के लिए Attack on Titan, Demon Slayer, या Jujutsu Kaisen देखें।"),
            ("मैं confused हूं", "No problem! मैं clarify करता हूं। आपको किस बारे में confusion है?"),
            ("Best animation quality किसमें है?", "Best animation के लिए Demon Slayer (Ufotable), Violet Evergarden (KyoAni), या Mob Psycho 100 (Bones) देखें।"),
        ]
        new_data.extend([{'input': i, 'output': o, 'category': 'code_switching'} for i, o in more_code_switching])
        
        # Combine and write
        all_data = existing_data + new_data
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['input', 'output', 'category'])
            writer.writeheader()
            writer.writerows(all_data)
        
        print(f"✓ Expanded to {len(all_data)} examples (added {len(new_data)})")
    
    def expand_anti_hallucination_dataset(self):
        """Expand anti-hallucination dataset to 500+ examples"""
        print("\n[4/4] Expanding Anti-Hallucination Dataset...")
        
        file_path = self.output_dir / "anti_hallucination_dataset.csv"
        
        # Read existing
        existing_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_data = list(reader)
        
        new_data = []
        
        # Add more factual corrections
        factual_corrections = [
            ("Is Goku from Naruto?", "No, Goku is from Dragon Ball, not Naruto. Naruto's main character is Naruto Uzumaki."),
            ("Did Naruto create anime?", "No, Naruto is an anime series created by Masashi Kishimoto. Anime as a medium has existed since the early 1900s."),
            ("Is Attack on Titan a comedy?", "No, Attack on Titan is primarily a dark fantasy action series with serious themes, though it has occasional lighter moments."),
            ("Are all anime violent?", "No, anime spans many genres. While some contain violence, many are peaceful slice-of-life, romance, or comedy series."),
            ("Is Studio Ghibli the only anime studio?", "No, there are many anime studios including Kyoto Animation, MAPPA, Ufotable, Bones, and many others."),
            ("Did One Piece end?", "No, One Piece is still ongoing with new manga chapters and anime episodes being released regularly."),
            ("Is anime only from Japan?", "Anime specifically refers to Japanese animation. Similar styles from other countries are typically called by different names."),
            ("Are manga and anime the same?", "No, manga are Japanese comics/graphic novels, while anime are animated shows. Many anime are adapted from manga."),
            ("Is every anime based on manga?", "No, some anime are original works, while others are adapted from light novels, video games, or visual novels."),
            ("Did Hayao Miyazaki make Naruto?", "No, Hayao Miyazaki is a Studio Ghibli director. Naruto was created by Masashi Kishimoto."),
        ]
        new_data.extend([{'input': i, 'output': o, 'category': 'factual'} for i, o in factual_corrections])
        
        # Add more uncertainty acknowledgments
        more_uncertainty = [
            ("What will happen next in One Piece?", "I can't predict future plot developments. I can only discuss what has been officially released."),
            ("Who would win: Goku or Saitama?", "That's a hypothetical question with no definitive answer since they're from different universes with different rules."),
            ("What's the meaning of life in anime?", "Different anime explore different philosophical themes. There's no single answer, as each series presents its own perspective."),
            ("Is this the best anime ever?", "Best is subjective and depends on personal preferences. What one person loves, another might not enjoy."),
            ("Will I like this anime?", "I can't guarantee you'll like any specific anime, but I can recommend based on your stated preferences and similar series you've enjoyed."),
            ("What's your favorite character?", "As an AI, I don't have personal favorites or preferences. I can discuss popular characters or help you find characters you might like."),
            ("Should I watch sub or dub?", "That's a personal choice. Both have advantages - subs preserve original performances, dubs offer convenience. Try both and see what you prefer."),
            ("Is anime better than movies?", "That's subjective. Anime and movies are different mediums, each with unique strengths. Enjoyment depends on personal taste."),
            ("What anime will make me cry?", "Emotional responses vary by person. I can recommend anime known for emotional content, but whether it affects you depends on your sensibilities."),
            ("Which studio makes the best anime?", "Different studios excel in different areas. The 'best' depends on what aspects you value most in animation and storytelling."),
        ]
        new_data.extend([{'input': i, 'output': o, 'category': 'uncertainty'} for i, o in more_uncertainty])
        
        # Combine and write
        all_data = existing_data + new_data
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['input', 'output', 'category'])
            writer.writeheader()
            writer.writerows(all_data)
        
        print(f"✓ Expanded to {len(all_data)} examples (added {len(new_data)})")
    
    def count_total_examples(self):
        """Count total examples across all datasets"""
        total = 0
        for filename in ['conversational_dataset.csv', 'anime_dataset.csv', 
                        'hindi_english_dataset.csv', 'anti_hallucination_dataset.csv']:
            file_path = self.output_dir / filename
            with open(file_path, 'r', encoding='utf-8') as f:
                total += sum(1 for _ in f) - 1  # Subtract header
        return total


def main():
    """Main execution"""
    expander = DatasetExpander(output_dir="data/raw")
    expander.expand_all_datasets()


if __name__ == "__main__":
    main()
