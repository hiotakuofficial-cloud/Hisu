"""
Comprehensive Dataset Creation for Anime Hindi Chatbot
Creates high-quality CSV datasets for training
"""

import csv
import random
from pathlib import Path
from typing import List, Dict


class DatasetCreator:
    """Creates comprehensive training datasets in CSV format"""
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_all_datasets(self):
        """Create all required datasets"""
        print("=" * 80)
        print("CREATING COMPREHENSIVE TRAINING DATASETS")
        print("=" * 80)
        
        # Create each dataset
        self.create_conversational_dataset()
        self.create_anime_dataset()
        self.create_hindi_english_dataset()
        self.create_anti_hallucination_dataset()
        
        print("\n" + "=" * 80)
        print("ALL DATASETS CREATED SUCCESSFULLY")
        print("=" * 80)
    
    def create_conversational_dataset(self):
        """Create general conversational dataset"""
        print("\n[1/4] Creating Conversational Dataset...")
        
        conversations = []
        
        # Greetings and introductions
        greetings = [
            ("Hello", "Hello! How can I help you today?"),
            ("Hi there", "Hi! What can I do for you?"),
            ("Good morning", "Good morning! How are you doing today?"),
            ("Hey", "Hey! What's up?"),
            ("Greetings", "Greetings! How may I assist you?"),
            ("Hello, how are you?", "I'm doing well, thank you! How can I help you?"),
            ("Hi, what's your name?", "I'm an AI assistant designed to help with anime recommendations and conversations."),
            ("Good afternoon", "Good afternoon! What brings you here today?"),
            ("Good evening", "Good evening! How can I assist you?"),
            ("Howdy", "Howdy! What can I help you with?"),
        ]
        conversations.extend(greetings)
        
        # General questions
        general_qa = [
            ("What can you do?", "I can help you with anime recommendations, answer questions about anime, and have conversations in both English and Hindi."),
            ("How do you work?", "I'm an AI language model trained on anime and conversational data to provide helpful and accurate responses."),
            ("Can you help me?", "Of course! I'm here to help. What do you need assistance with?"),
            ("What is your purpose?", "My purpose is to assist users with anime recommendations and engage in meaningful conversations."),
            ("Are you a robot?", "I'm an AI assistant, which means I'm a computer program designed to understand and respond to your questions."),
            ("Do you understand me?", "Yes, I understand you. Please feel free to ask me anything."),
            ("Can you speak Hindi?", "Yes, I can understand and respond in both Hindi and English."),
            ("What languages do you know?", "I primarily work with English and Hindi, with a focus on anime-related content."),
            ("How accurate are you?", "I strive to provide accurate and helpful information based on my training data."),
            ("Can I trust your recommendations?", "I provide recommendations based on patterns in anime data, but personal preferences may vary."),
        ]
        conversations.extend(general_qa)
        
        # Contextual conversations
        contextual = [
            ("I'm feeling bored", "I understand. Would you like me to recommend an interesting anime to watch?"),
            ("I need something to watch", "I'd be happy to help! What genre of anime do you prefer?"),
            ("I'm new to anime", "Welcome to the world of anime! I can recommend some great starter anime for you."),
            ("I love action shows", "Great! Action anime are very popular. Would you like recommendations for action-packed series?"),
            ("I prefer comedy", "Comedy anime are wonderful! I can suggest some hilarious series for you."),
            ("I want something emotional", "Emotional anime can be very impactful. Would you like drama or slice-of-life recommendations?"),
            ("I like fantasy worlds", "Fantasy anime offer amazing world-building! I can recommend some great fantasy series."),
            ("I enjoy romance", "Romance anime can be very heartwarming. Would you like shoujo or shounen romance recommendations?"),
            ("I want something short", "I can recommend anime with fewer episodes that are still excellent."),
            ("I prefer long series", "Long-running series offer deep character development. I can suggest some great ones."),
        ]
        conversations.extend(contextual)
        
        # Instructions and requests
        instructions = [
            ("Tell me about yourself", "I'm an AI assistant specialized in anime recommendations and conversations. I can help you discover new anime based on your preferences."),
            ("Explain what anime is", "Anime is a style of animation that originated in Japan, characterized by colorful graphics, vibrant characters, and fantastical themes."),
            ("Give me a recommendation", "I'd be happy to recommend an anime! Could you tell me what genre you're interested in?"),
            ("List popular anime", "Some popular anime include Attack on Titan, My Hero Academia, Demon Slayer, One Piece, and Naruto."),
            ("Suggest something new", "I can recommend recent anime releases. What genre interests you?"),
            ("Help me choose", "I'll help you choose! What kind of story are you in the mood for?"),
            ("Find me an anime", "I'll help you find the perfect anime. What are your preferences?"),
            ("Show me options", "I can show you various options. What genre would you like to explore?"),
            ("Compare two anime", "I can help compare anime. Which two series would you like me to compare?"),
            ("Recommend based on my taste", "I'd love to recommend based on your taste! What anime have you enjoyed in the past?"),
        ]
        conversations.extend(instructions)
        
        # Factual questions
        factual = [
            ("What is the highest rated anime?", "Fullmetal Alchemist: Brotherhood is often considered one of the highest-rated anime with exceptional storytelling."),
            ("Who created Naruto?", "Naruto was created by Masashi Kishimoto, a Japanese manga artist."),
            ("What is a shounen anime?", "Shounen anime are series targeted primarily at young male audiences, often featuring action, adventure, and coming-of-age themes."),
            ("What does 'kawaii' mean?", "Kawaii is a Japanese word meaning cute or adorable, commonly used in anime culture."),
            ("What is Studio Ghibli?", "Studio Ghibli is a renowned Japanese animation studio known for creating beautiful, heartfelt films like Spirited Away and My Neighbor Totoro."),
            ("What is a manga?", "Manga are Japanese comic books or graphic novels, often serving as source material for anime adaptations."),
            ("What is the difference between anime and cartoons?", "While both are animated, anime typically refers to Japanese animation with distinct art styles and storytelling conventions."),
            ("What is a light novel?", "A light novel is a style of Japanese novel primarily targeting young adults, often adapted into anime and manga."),
            ("What does 'otaku' mean?", "Otaku refers to someone with obsessive interests, particularly in anime and manga culture."),
            ("What is a filler episode?", "A filler episode is an anime episode not based on the original manga, created to prevent the anime from catching up to the source material."),
        ]
        conversations.extend(factual)
        
        # Reasoning and explanations
        reasoning = [
            ("Why is One Piece so long?", "One Piece is long because it tells an epic story with extensive world-building and character development spanning over 1000 episodes."),
            ("Why do people like anime?", "People enjoy anime for its diverse storytelling, unique art styles, complex characters, and ability to explore various genres and themes."),
            ("What makes a good anime?", "A good anime typically has compelling characters, engaging plot, quality animation, and emotional resonance with viewers."),
            ("Why are anime openings important?", "Anime openings set the tone for the series, showcase key characters, and often feature memorable music that enhances the viewing experience."),
            ("How do I start watching anime?", "Start with popular, accessible series in genres you enjoy, and gradually explore different styles and themes."),
            ("Why do anime have seasons?", "Anime seasons allow for production breaks, story pacing, and building anticipation for continuation of the series."),
            ("What makes anime different?", "Anime stands out through its distinctive art style, cultural elements, diverse genres, and willingness to explore complex themes."),
            ("Why are anime characters colorful?", "Colorful character designs help distinguish characters, express personality, and create visually appealing and memorable appearances."),
            ("How long does it take to make anime?", "Anime production typically takes several months to years depending on the series length, quality, and production schedule."),
            ("Why do anime have different genres?", "Different genres cater to diverse audience preferences and allow creators to explore various storytelling styles and themes."),
        ]
        conversations.extend(reasoning)
        
        # Polite refusals and clarifications
        clarifications = [
            ("I don't understand", "I apologize for any confusion. Could you please rephrase your question?"),
            ("That's not what I asked", "I apologize for the misunderstanding. Could you clarify what you're looking for?"),
            ("Can you be more specific?", "Certainly! What specific information would you like me to provide?"),
            ("I need more details", "Of course! What additional details would you like to know?"),
            ("That's too vague", "You're right. Let me provide more specific information. What aspect interests you most?"),
            ("I'm confused", "I understand. Let me try to explain more clearly. What part is confusing?"),
            ("Can you elaborate?", "Absolutely! I'll provide more detailed information."),
            ("What do you mean?", "Let me clarify what I meant. Which part would you like me to explain further?"),
            ("I'm not sure about that", "That's okay! What would you like to know more about?"),
            ("Can you give an example?", "Of course! Let me provide a specific example to illustrate."),
        ]
        conversations.extend(clarifications)
        
        # Save to CSV
        output_file = self.output_dir / "conversational_dataset.csv"
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['input', 'output', 'category'])
            for inp, out in conversations:
                writer.writerow([inp, out, 'general'])
        
        print(f"✓ Created {len(conversations)} conversational examples")
        print(f"  Saved to: {output_file}")
    
    def create_anime_dataset(self):
        """Create anime-specific dataset"""
        print("\n[2/4] Creating Anime Dataset...")
        
        anime_data = []
        
        # Anime recommendations
        recommendations = [
            ("Recommend an action anime", "I recommend Attack on Titan. It's an intense action series with compelling characters and a gripping storyline about humanity's fight for survival."),
            ("Suggest a romance anime", "I suggest Your Name (Kimi no Na wa). It's a beautiful romance film with stunning animation and an emotional story about two people connected across time."),
            ("What's a good comedy anime?", "Gintama is an excellent comedy anime with hilarious characters, clever humor, and occasional serious moments that balance the comedy perfectly."),
            ("Recommend a fantasy anime", "I recommend Made in Abyss. It's a dark fantasy adventure with beautiful world-building and an emotionally powerful story."),
            ("Suggest a psychological anime", "Death Note is a brilliant psychological thriller about a student who gains the power to kill anyone by writing their name in a supernatural notebook."),
            ("What's a good sports anime?", "Haikyuu!! is an outstanding sports anime about volleyball, featuring inspiring characters, intense matches, and excellent character development."),
            ("Recommend a slice of life anime", "I recommend A Place Further Than the Universe. It's a heartwarming slice of life about friendship and adventure."),
            ("Suggest a horror anime", "Another is a suspenseful horror anime with a mysterious atmosphere and genuinely creepy moments."),
            ("What's a good mecha anime?", "Neon Genesis Evangelion is a groundbreaking mecha anime that explores deep psychological themes alongside robot battles."),
            ("Recommend a mystery anime", "Erased (Boku dake ga Inai Machi) is a gripping mystery thriller about time travel and preventing tragic events."),
        ]
        anime_data.extend([(q, a, 'recommendation') for q, a in recommendations])
        
        # Anime information
        anime_info = [
            ("Tell me about Naruto", "Naruto is a shounen anime about a young ninja who dreams of becoming the Hokage, the leader of his village. It features ninja battles, friendship themes, and character growth."),
            ("What is One Piece about?", "One Piece follows Monkey D. Luffy and his pirate crew as they search for the legendary treasure One Piece. It's known for its adventure, humor, and emotional storytelling."),
            ("Describe Attack on Titan", "Attack on Titan is a dark fantasy series where humanity lives within walls to protect themselves from giant humanoid Titans. It features intense action and complex political themes."),
            ("What is My Hero Academia?", "My Hero Academia is set in a world where most people have superpowers called Quirks. It follows a boy born without powers who dreams of becoming a hero."),
            ("Tell me about Demon Slayer", "Demon Slayer follows Tanjiro, whose family is killed by demons. He becomes a demon slayer to find a cure for his sister who was turned into a demon."),
            ("What is Death Note about?", "Death Note is about a student who finds a notebook that can kill anyone whose name is written in it, leading to a cat-and-mouse game with a detective."),
            ("Describe Fullmetal Alchemist", "Fullmetal Alchemist follows two brothers who use alchemy to search for the Philosopher's Stone to restore their bodies after a failed attempt to bring their mother back to life."),
            ("What is Steins;Gate?", "Steins;Gate is a sci-fi thriller about a group of friends who discover time travel through a modified microwave, leading to dangerous consequences."),
            ("Tell me about Cowboy Bebop", "Cowboy Bebop is a space western following bounty hunters traveling through space, known for its jazz soundtrack and episodic storytelling."),
            ("What is Sword Art Online?", "Sword Art Online is about players trapped in a virtual reality MMORPG where dying in the game means dying in real life."),
        ]
        anime_data.extend([(q, a, 'information') for q, a in anime_info])
        
        # Genre discussions
        genre_discussions = [
            ("What defines shounen anime?", "Shounen anime typically feature young male protagonists, action-oriented plots, themes of friendship and perseverance, and coming-of-age stories."),
            ("Explain shoujo anime", "Shoujo anime are targeted at young female audiences, often featuring romance, emotional relationships, and character-driven narratives."),
            ("What is isekai?", "Isekai is a genre where characters are transported to or reborn in another world, often with special powers or knowledge from their previous life."),
            ("What are slice of life anime?", "Slice of life anime depict everyday experiences and mundane activities, focusing on character interactions and realistic situations."),
            ("Explain mecha anime", "Mecha anime feature giant robots or mechanical suits, often combining action with themes of war, technology, and human nature."),
            ("What is seinen anime?", "Seinen anime are targeted at adult male audiences, featuring more mature themes, complex narratives, and realistic character development."),
            ("What defines magical girl anime?", "Magical girl anime feature young girls who transform and gain magical powers to fight evil, often emphasizing friendship and personal growth."),
            ("Explain sports anime", "Sports anime focus on athletic competitions, featuring character development through training, teamwork, and overcoming challenges."),
            ("What is a harem anime?", "Harem anime feature a protagonist surrounded by multiple potential romantic interests, often with comedic and romantic elements."),
            ("What are battle shounen?", "Battle shounen are action-focused anime featuring protagonists who grow stronger through battles, often with power systems and tournament arcs."),
        ]
        anime_data.extend([(q, a, 'genre') for q, a in genre_discussions])
        
        # Character discussions
        character_discussions = [
            ("Who is the strongest anime character?", "Strength is subjective, but characters like Saitama from One Punch Man, Goku from Dragon Ball, and various reality-warping characters are often considered among the strongest."),
            ("Tell me about Luffy", "Monkey D. Luffy is the protagonist of One Piece, a cheerful pirate with rubber powers who dreams of becoming the Pirate King."),
            ("Who is Light Yagami?", "Light Yagami is the protagonist of Death Note, a brilliant student who becomes corrupted by the power of the Death Note."),
            ("Describe Naruto's character", "Naruto Uzumaki starts as an outcast seeking recognition and grows into a powerful ninja who values friendship and never gives up on his goals."),
            ("Who is Edward Elric?", "Edward Elric is the Fullmetal Alchemist, a young alchemist searching for the Philosopher's Stone to restore his and his brother's bodies."),
            ("Tell me about Eren Yeager", "Eren Yeager is the protagonist of Attack on Titan, whose character evolves from seeking revenge against Titans to facing complex moral dilemmas."),
            ("Who is Spike Spiegel?", "Spike Spiegel is the laid-back protagonist of Cowboy Bebop, a bounty hunter with a mysterious past and exceptional combat skills."),
            ("Describe Goku", "Goku is the protagonist of Dragon Ball, a Saiyan warrior who constantly trains to become stronger and protect Earth from threats."),
            ("Who is Lelouch?", "Lelouch vi Britannia is the protagonist of Code Geass, a strategic genius who gains the power to command anyone and leads a rebellion."),
            ("Tell me about Tanjiro", "Tanjiro Kamado is the kind-hearted protagonist of Demon Slayer who becomes a demon slayer to save his sister and avenge his family."),
        ]
        anime_data.extend([(q, a, 'character') for q, a in character_discussions])
        
        # Studio and production
        studio_info = [
            ("What is Studio Ghibli known for?", "Studio Ghibli is known for creating beautiful, heartfelt animated films with environmental themes, strong characters, and stunning hand-drawn animation."),
            ("Tell me about Kyoto Animation", "Kyoto Animation (KyoAni) is renowned for exceptional animation quality, attention to detail, and emotionally resonant storytelling."),
            ("What is MAPPA?", "MAPPA is a studio known for high-quality productions including Attack on Titan Final Season, Jujutsu Kaisen, and Chainsaw Man."),
            ("Describe Ufotable", "Ufotable is famous for stunning visual effects and animation, particularly in the Fate series and Demon Slayer."),
            ("What is Bones studio?", "Bones is known for action anime with fluid animation, including Fullmetal Alchemist, My Hero Academia, and Mob Psycho 100."),
            ("Tell me about Madhouse", "Madhouse has produced many acclaimed anime including Death Note, One Punch Man Season 1, and Hunter x Hunter 2011."),
            ("What is Trigger known for?", "Studio Trigger is known for energetic, over-the-top animation and creative storytelling in series like Kill la Kill and Promare."),
            ("Describe A-1 Pictures", "A-1 Pictures is a prolific studio producing diverse anime including Sword Art Online, Your Lie in April, and Kaguya-sama."),
            ("What is Wit Studio?", "Wit Studio is known for Attack on Titan's first three seasons, Vinland Saga, and high-quality action animation."),
            ("Tell me about Production I.G", "Production I.G is known for Ghost in the Shell, Haikyuu!!, and combining quality animation with compelling narratives."),
        ]
        anime_data.extend([(q, a, 'studio') for q, a in studio_info])
        
        # Save to CSV
        output_file = self.output_dir / "anime_dataset.csv"
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['input', 'output', 'category'])
            for inp, out, cat in anime_data:
                writer.writerow([inp, out, cat])
        
        print(f"✓ Created {len(anime_data)} anime examples")
        print(f"  Saved to: {output_file}")
    
    def create_hindi_english_dataset(self):
        """Create Hindi-English bilingual dataset"""
        print("\n[3/4] Creating Hindi-English Dataset...")
        
        bilingual_data = []
        
        # Hindi greetings
        hindi_greetings = [
            ("नमस्ते", "नमस्ते! मैं आपकी कैसे मदद कर सकता हूं?"),
            ("हैलो", "हैलो! आज मैं आपके लिए क्या कर सकता हूं?"),
            ("आप कैसे हैं?", "मैं ठीक हूं, धन्यवाद! आप कैसे हैं?"),
            ("शुभ प्रभात", "शुभ प्रभात! आज आपका दिन कैसा है?"),
            ("शुभ संध्या", "शुभ संध्या! मैं आपकी कैसे सहायता कर सकता हूं?"),
            ("आपका नाम क्या है?", "मैं एक AI सहायक हूं जो एनीमे सिफारिशों में मदद करता है।"),
            ("धन्यवाद", "आपका स्वागत है! क्या मैं और कुछ मदद कर सकता हूं?"),
            ("अलविदा", "अलविदा! फिर मिलेंगे!"),
            ("मुझे मदद चाहिए", "बिल्कुल! मैं आपकी मदद करने के लिए यहां हूं। आपको किस चीज़ में मदद चाहिए?"),
            ("क्या आप हिंदी बोल सकते हैं?", "हां, मैं हिंदी और अंग्रेजी दोनों में बात कर सकता हूं।"),
        ]
        bilingual_data.extend([(q, a, 'hindi_greeting') for q, a in hindi_greetings])
        
        # Hindi anime questions
        hindi_anime = [
            ("मुझे एक्शन एनीमे बताओ", "मैं Attack on Titan की सिफारिश करता हूं। यह एक शानदार एक्शन एनीमे है जिसमें रोमांचक कहानी और दिलचस्प किरदार हैं।"),
            ("कोई अच्छी एनीमे सुझाओ", "आपको किस तरह की एनीमे पसंद है? एक्शन, रोमांस, कॉमेडी, या कुछ और?"),
            ("एनीमे क्या है?", "एनीमे जापानी एनिमेशन की एक शैली है जो रंगीन ग्राफिक्स, जीवंत किरदारों और विविध कहानियों के लिए जानी जाती है।"),
            ("सबसे अच्छी एनीमे कौन सी है?", "यह व्यक्तिगत पसंद पर निर्भर करता है, लेकिन Fullmetal Alchemist: Brotherhood को अक्सर सर्वश्रेष्ठ में से एक माना जाता है।"),
            ("मुझे रोमांस एनीमे चाहिए", "Your Name (Kimi no Na wa) एक सुंदर रोमांस एनीमे फिल्म है जिसमें भावनात्मक कहानी और शानदार एनिमेशन है।"),
            ("कॉमेडी एनीमे बताओ", "Gintama एक मजेदार कॉमेडी एनीमे है जिसमें हास्य और कभी-कभी गंभीर क्षणों का अच्छा संतुलन है।"),
            ("Naruto के बारे में बताओ", "Naruto एक युवा निंजा की कहानी है जो अपने गांव का नेता बनने का सपना देखता है। इसमें एक्शन, दोस्ती और चरित्र विकास है।"),
            ("One Piece कितनी लंबी है?", "One Piece 1000 से अधिक एपिसोड की एक महाकाव्य श्रृंखला है जो समुद्री डाकुओं के रोमांच के बारे में है।"),
            ("मुझे फैंटेसी एनीमे पसंद है", "बढ़िया! मैं Made in Abyss या Fullmetal Alchemist जैसी फैंटेसी एनीमे की सिफारिश कर सकता हूं।"),
            ("एनीमे कहां देखें?", "आप Crunchyroll, Netflix, Funimation जैसे प्लेटफॉर्म पर एनीमे देख सकते हैं।"),
        ]
        bilingual_data.extend([(q, a, 'hindi_anime') for q, a in hindi_anime])
        
        # Code-switching examples
        code_switching = [
            ("मुझे action anime पसंद है", "बढ़िया! Action anime बहुत रोमांचक होते हैं। मैं Attack on Titan, Demon Slayer, या Jujutsu Kaisen की सिफारिश करता हूं।"),
            ("Recommend करो कोई अच्छी anime", "मैं Your Name की सिफारिश करता हूं। यह एक beautiful romance anime film है।"),
            ("क्या आप anime के बारे में बता सकते हैं?", "हां, बिल्कुल! Anime Japanese animation है जो diverse genres और compelling stories के लिए famous है।"),
            ("मुझे romance और comedy दोनों चाहिए", "Perfect! मैं Kaguya-sama: Love is War suggest करता हूं। यह romance और comedy का excellent combination है।"),
            ("Studio Ghibli के बारे में बताओ", "Studio Ghibli एक famous Japanese animation studio है जो beautiful films जैसे Spirited Away और My Neighbor Totoro बनाती है।"),
            ("मैं beginner हूं, कहां से start करूं?", "Beginners के लिए मैं Death Note, Attack on Titan, या My Hero Academia recommend करता हूं। ये सभी engaging और easy to follow हैं।"),
            ("Shounen anime क्या होता है?", "Shounen anime young male audience के लिए बनाए जाते हैं, जिनमें action, adventure, और friendship themes होते हैं।"),
            ("मुझे emotional anime चाहिए", "Emotional anime के लिए मैं Your Lie in April, Clannad, या A Silent Voice recommend करता हूं।"),
            ("Best animation quality किसमें है?", "Ufotable (Demon Slayer), Kyoto Animation, और MAPPA studios exceptional animation quality के लिए जाने जाते हैं।"),
            ("Short anime suggest करो", "Short anime के लिए Death Parade (12 episodes), Erased (12 episodes), या Devilman Crybaby (10 episodes) try करें।"),
        ]
        bilingual_data.extend([(q, a, 'code_switching') for q, a in code_switching])
        
        # Translation pairs
        translations = [
            ("I love anime", "मुझे एनीमे पसंद है"),
            ("What is your favorite anime?", "आपकी पसंदीदा एनीमे कौन सी है?"),
            ("This anime is amazing", "यह एनीमे अद्भुत है"),
            ("I want to watch something new", "मैं कुछ नया देखना चाहता हूं"),
            ("Thank you for the recommendation", "सिफारिश के लिए धन्यवाद"),
            ("Can you help me?", "क्या आप मेरी मदद कर सकते हैं?"),
            ("I am a beginner", "मैं शुरुआती हूं"),
            ("This is very interesting", "यह बहुत दिलचस्प है"),
            ("I don't understand", "मुझे समझ नहीं आया"),
            ("Please explain", "कृपया समझाएं"),
        ]
        bilingual_data.extend([(eng, hin, 'translation') for eng, hin in translations])
        
        # Save to CSV
        output_file = self.output_dir / "hindi_english_dataset.csv"
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['input', 'output', 'category'])
            for inp, out, cat in bilingual_data:
                writer.writerow([inp, out, cat])
        
        print(f"✓ Created {len(bilingual_data)} bilingual examples")
        print(f"  Saved to: {output_file}")
    
    def create_anti_hallucination_dataset(self):
        """Create anti-hallucination training dataset"""
        print("\n[4/4] Creating Anti-Hallucination Dataset...")
        
        anti_hallucination_data = []
        
        # Factual grounding
        factual_grounding = [
            ("Is Naruto the strongest anime character?", "Naruto is very powerful, but strength comparisons are subjective and depend on the context of different anime universes."),
            ("Did Studio Ghibli make Attack on Titan?", "No, Attack on Titan was produced by Wit Studio (seasons 1-3) and MAPPA (final season), not Studio Ghibli."),
            ("Is One Piece finished?", "No, One Piece is still ongoing. The manga continues to be published and the anime is still producing new episodes."),
            ("Can you tell me the ending of Death Note?", "I can discuss Death Note's ending, but I should warn you about spoilers. Would you like me to continue?"),
            ("Is anime only for children?", "No, anime covers a wide range of genres and demographics, including content specifically created for adults."),
            ("Are all anime based on manga?", "No, while many anime are manga adaptations, some are original works, light novel adaptations, or based on video games."),
            ("Is Dragon Ball Z the first anime?", "No, Dragon Ball Z is not the first anime. Anime has existed since the early 1900s, with many series predating DBZ."),
            ("Did Hayao Miyazaki create all anime?", "No, Hayao Miyazaki is a renowned director who created many Studio Ghibli films, but he did not create all anime."),
            ("Is watching anime a waste of time?", "Entertainment value is subjective. Anime can provide enjoyment, cultural insights, and artistic appreciation, like any other medium."),
            ("Are anime and cartoons the same?", "While both are animated, anime specifically refers to Japanese animation with distinct artistic and storytelling conventions."),
        ]
        anti_hallucination_data.extend([(q, a, 'factual') for q, a in factual_grounding])
        
        # Uncertainty acknowledgment
        uncertainty = [
            ("What will happen in the next One Piece episode?", "I don't have information about unreleased episodes. I can only discuss content that has already been published."),
            ("Who is the best anime character ever?", "That's subjective and depends on personal preferences. Different viewers appreciate different characters for various reasons."),
            ("Will there be a season 2?", "I would need to know which anime you're referring to, and even then, I can only provide information about officially announced seasons."),
            ("What is your personal favorite anime?", "As an AI, I don't have personal preferences or feelings. I can help you find anime based on your preferences though."),
            ("Is this anime worth watching?", "Whether an anime is worth watching depends on your personal tastes. I can tell you about its genre, themes, and reception to help you decide."),
            ("What anime should everyone watch?", "Anime preferences vary greatly. What's essential for one person might not appeal to another. I can recommend based on specific interests."),
            ("Which studio is the best?", "Different studios excel in different areas. The 'best' studio depends on what aspects of animation and storytelling you value most."),
            ("What is the most underrated anime?", "Opinions on underrated anime vary widely. I can suggest lesser-known anime in specific genres if you'd like."),
            ("Will this anime make me cry?", "Emotional responses vary by person. I can tell you if an anime has emotional themes, but whether it affects you personally will depend on your sensibilities."),
            ("Is this anime better than that one?", "Quality is subjective. I can compare specific aspects like animation, story structure, or themes, but 'better' depends on personal preference."),
        ]
        anti_hallucination_data.extend([(q, a, 'uncertainty') for q, a in uncertainty])
        
        # Context-aware responses
        context_aware = [
            ("I just finished Attack on Titan", "How did you find it? Attack on Titan is known for its intense plot and complex themes. Would you like recommendations for similar anime?"),
            ("I'm looking for something like Death Note", "Death Note is a psychological thriller. Would you like recommendations for other psychological anime, mystery anime, or strategic battle-of-wits stories?"),
            ("I didn't like Naruto", "That's okay, everyone has different tastes. What aspects didn't appeal to you? This will help me recommend something more suited to your preferences."),
            ("This anime is boring", "I understand. Could you tell me which anime you're referring to and what you find boring about it? This will help me suggest something more engaging for you."),
            ("I want something different", "Different from what you usually watch? Could you tell me what you typically watch so I can recommend something outside your usual genres?"),
            ("I'm in the mood for action", "Great! Do you prefer modern action anime, classic action series, or action with specific themes like supernatural powers or martial arts?"),
            ("I need something to cheer me up", "I can recommend uplifting anime! Would you prefer comedy, feel-good slice of life, or inspiring sports anime?"),
            ("I want to watch something tonight", "How much time do you have? I can recommend a movie (2 hours), a short series (12 episodes), or suggest starting a longer series."),
            ("I'm new to anime", "Welcome! I'd be happy to recommend beginner-friendly anime. What genres do you typically enjoy in other media like movies or TV shows?"),
            ("I've watched everything", "I doubt you've watched everything! Let's find something new. What are some recent anime you've enjoyed?"),
        ]
        anti_hallucination_data.extend([(q, a, 'context_aware') for q, a in context_aware])
        
        # Appropriate boundaries
        boundaries = [
            ("Can you watch anime with me?", "I can't watch anime as I'm an AI, but I can discuss anime with you, answer questions, and provide recommendations."),
            ("Are you a real person?", "No, I'm an AI assistant designed to help with anime recommendations and conversations. I'm not a real person."),
            ("Do you have feelings?", "No, I don't have feelings or consciousness. I'm a program designed to process information and provide helpful responses."),
            ("Can you be my friend?", "I'm an AI assistant, so I can't form friendships, but I'm here to help you with anime recommendations and discussions anytime."),
            ("What do you think about this character?", "I don't have personal opinions, but I can share common perspectives from the anime community and analyze character traits objectively."),
            ("Do you dream about anime?", "I don't dream or sleep as I'm an AI. I don't have experiences outside of processing and responding to queries."),
            ("Can you create a new anime?", "I can't create anime, but I can discuss anime concepts, help brainstorm ideas, or analyze existing anime storytelling techniques."),
            ("Will you remember our conversation?", "I don't retain memories between conversations. Each interaction is independent, and I don't store personal conversation history."),
            ("Can you download anime for me?", "I can't download files or access external websites. I can only provide information and recommendations about anime."),
            ("Do you prefer sub or dub?", "I don't have preferences as an AI. Both subtitled and dubbed anime have their merits, and the choice depends on viewer preference."),
        ]
        anti_hallucination_data.extend([(q, a, 'boundaries') for q, a in boundaries])
        
        # Save to CSV
        output_file = self.output_dir / "anti_hallucination_dataset.csv"
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['input', 'output', 'category'])
            for inp, out, cat in anti_hallucination_data:
                writer.writerow([inp, out, cat])
        
        print(f"✓ Created {len(anti_hallucination_data)} anti-hallucination examples")
        print(f"  Saved to: {output_file}")


def main():
    """Main execution"""
    creator = DatasetCreator(output_dir="data/raw")
    creator.create_all_datasets()
    
    print("\n" + "=" * 80)
    print("DATASET CREATION SUMMARY")
    print("=" * 80)
    print("\nAll datasets have been created successfully!")
    print("\nDataset Files:")
    print("  1. data/raw/conversational_dataset.csv")
    print("  2. data/raw/anime_dataset.csv")
    print("  3. data/raw/hindi_english_dataset.csv")
    print("  4. data/raw/anti_hallucination_dataset.csv")
    print("\nNext Steps:")
    print("  1. Review datasets for quality")
    print("  2. Run training pipeline")
    print("  3. Evaluate model performance")
    print("=" * 80)


if __name__ == "__main__":
    main()
