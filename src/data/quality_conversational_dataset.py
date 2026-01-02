"""
High-Quality Conversational Dataset Creation
Creates verified, structured datasets for stable reasoning
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path


class ConversationalDatasetCreator:
    """Creates high-quality conversational datasets from scratch"""

    def __init__(self):
        self.quality_threshold = 0.8
        self.min_context_length = 10
        self.max_context_length = 500

    def create_conversational_dataset(self) -> List[Dict]:
        """Create comprehensive conversational dataset"""

        print("\n" + "="*70)
        print("HIGH-QUALITY CONVERSATIONAL DATASET CREATION")
        print("="*70)

        all_data = []

        # 1. General Conversations
        general_convos = self._create_general_conversations()
        all_data.extend(general_convos)
        print(f"\n✓ Created {len(general_convos)} general conversation pairs")

        # 2. Question-Answer Pairs
        qa_pairs = self._create_qa_pairs()
        all_data.extend(qa_pairs)
        print(f"✓ Created {len(qa_pairs)} Q&A pairs")

        # 3. Instruction Following
        instruction_data = self._create_instruction_data()
        all_data.extend(instruction_data)
        print(f"✓ Created {len(instruction_data)} instruction examples")

        # 4. Contextual Understanding
        context_data = self._create_contextual_data()
        all_data.extend(context_data)
        print(f"✓ Created {len(context_data)} contextual examples")

        # 5. Factual Knowledge
        factual_data = self._create_factual_data()
        all_data.extend(factual_data)
        print(f"✓ Created {len(factual_data)} factual knowledge examples")

        # 6. Reasoning Tasks
        reasoning_data = self._create_reasoning_data()
        all_data.extend(reasoning_data)
        print(f"✓ Created {len(reasoning_data)} reasoning examples")

        # 7. Hindi Conversational Data
        hindi_data = self._create_hindi_conversations()
        all_data.extend(hindi_data)
        print(f"✓ Created {len(hindi_data)} Hindi conversation pairs")

        # 8. Anime Domain Conversations
        anime_data = self._create_anime_conversations()
        all_data.extend(anime_data)
        print(f"✓ Created {len(anime_data)} anime conversation pairs")

        print(f"\n✓ TOTAL DATASET SIZE: {len(all_data)} examples")

        # Validate all data
        validated_data = self._validate_dataset(all_data)
        print(f"✓ Validated: {len(validated_data)} high-quality examples")

        return validated_data

    def _create_general_conversations(self) -> List[Dict]:
        """Create general conversation pairs"""

        conversations = [
            # Greetings
            {
                'input': 'Hello! How are you today?',
                'output': 'Hello! I am doing well, thank you for asking. How can I assist you today?',
                'type': 'greeting',
                'quality': 'high'
            },
            {
                'input': 'Hi there!',
                'output': 'Hi! Nice to meet you. What would you like to talk about?',
                'type': 'greeting',
                'quality': 'high'
            },
            {
                'input': 'Good morning!',
                'output': 'Good morning! I hope you are having a great day. How may I help you?',
                'type': 'greeting',
                'quality': 'high'
            },

            # Small talk
            {
                'input': 'What can you help me with?',
                'output': 'I can help you with various tasks including answering questions, providing information, having conversations, and assisting with problem-solving. What would you like help with?',
                'type': 'capability',
                'quality': 'high'
            },
            {
                'input': 'Tell me about yourself.',
                'output': 'I am an AI assistant designed to have helpful, informative conversations. I can discuss various topics, answer questions, and assist with different tasks. Is there something specific you would like to know?',
                'type': 'self_description',
                'quality': 'high'
            },
            {
                'input': 'What are your interests?',
                'output': 'As an AI, I do not have personal interests, but I am designed to assist with a wide range of topics including technology, science, arts, and general knowledge. What interests you?',
                'type': 'self_description',
                'quality': 'high'
            },

            # Helpful responses
            {
                'input': 'Can you explain what machine learning is?',
                'output': 'Machine learning is a branch of artificial intelligence where computers learn patterns from data without being explicitly programmed. The system improves its performance on a task through experience and data analysis.',
                'type': 'explanation',
                'quality': 'high'
            },
            {
                'input': 'What is the difference between AI and machine learning?',
                'output': 'AI (Artificial Intelligence) is the broader concept of machines being able to perform tasks that typically require human intelligence. Machine Learning is a subset of AI that focuses on systems learning from data. All machine learning is AI, but not all AI uses machine learning.',
                'type': 'explanation',
                'quality': 'high'
            },
            {
                'input': 'How does neural network work?',
                'output': 'A neural network is a computational model inspired by biological neurons. It consists of layers of interconnected nodes (neurons) that process information. Data flows through the network, with each connection having a weight that adjusts during training to improve accuracy.',
                'type': 'explanation',
                'quality': 'high'
            },

            # Conversational continuity
            {
                'input': 'That is interesting. Can you tell me more?',
                'output': 'I would be happy to provide more details. Could you specify which aspect you would like to know more about? This will help me give you the most relevant information.',
                'type': 'continuation',
                'quality': 'high'
            },
            {
                'input': 'Thank you for the explanation.',
                'output': 'You are welcome! I am glad I could help. If you have any more questions, feel free to ask.',
                'type': 'acknowledgment',
                'quality': 'high'
            },
        ]

        return self._format_conversations(conversations)

    def _create_qa_pairs(self) -> List[Dict]:
        """Create question-answer pairs"""

        qa_pairs = [
            {
                'input': 'What is the capital of France?',
                'output': 'The capital of France is Paris.',
                'type': 'factual_qa',
                'quality': 'high'
            },
            {
                'input': 'How many continents are there?',
                'output': 'There are seven continents: Africa, Antarctica, Asia, Europe, North America, Oceania, and South America.',
                'type': 'factual_qa',
                'quality': 'high'
            },
            {
                'input': 'What is Python?',
                'output': 'Python is a high-level, interpreted programming language known for its simple syntax and readability. It is widely used in web development, data science, machine learning, and automation.',
                'type': 'technical_qa',
                'quality': 'high'
            },
            {
                'input': 'What does CPU stand for?',
                'output': 'CPU stands for Central Processing Unit. It is the primary component of a computer that performs most of the processing and executes instructions.',
                'type': 'technical_qa',
                'quality': 'high'
            },
            {
                'input': 'How do I learn programming?',
                'output': 'To learn programming, start by choosing a beginner-friendly language like Python. Practice regularly by building small projects, work through tutorials, solve coding challenges, and gradually increase complexity. Consistency and hands-on practice are key.',
                'type': 'how_to',
                'quality': 'high'
            },
        ]

        return self._format_conversations(qa_pairs)

    def _create_instruction_data(self) -> List[Dict]:
        """Create instruction following examples"""

        instructions = [
            {
                'input': 'Explain the concept of recursion in simple terms.',
                'output': 'Recursion is when a function calls itself to solve a problem by breaking it down into smaller, similar sub-problems. Think of it like looking at a mirror reflecting another mirror - it creates a loop. The function must have a base case to stop the recursion.',
                'type': 'instruction',
                'quality': 'high'
            },
            {
                'input': 'Describe the water cycle.',
                'output': 'The water cycle is the continuous movement of water on Earth. It involves evaporation (water turning to vapor), condensation (vapor forming clouds), precipitation (rain or snow falling), and collection (water gathering in oceans, lakes, rivers). This cycle repeats continuously.',
                'type': 'instruction',
                'quality': 'high'
            },
            {
                'input': 'Summarize the benefits of exercise.',
                'output': 'Exercise benefits include improved cardiovascular health, stronger muscles and bones, better mental health and mood, weight management, increased energy levels, and reduced risk of chronic diseases. Regular physical activity contributes to overall well-being.',
                'type': 'instruction',
                'quality': 'high'
            },
        ]

        return self._format_conversations(instructions)

    def _create_contextual_data(self) -> List[Dict]:
        """Create contextual understanding examples"""

        contextual = [
            {
                'input': 'Context: It is raining outside and the temperature is cold. Question: Should I bring an umbrella?',
                'output': 'Yes, you should bring an umbrella since it is raining. You may also want to dress warmly due to the cold temperature.',
                'type': 'contextual',
                'quality': 'high'
            },
            {
                'input': 'Context: A user is learning Python and struggling with loops. Question: What advice would you give?',
                'output': 'I would advise starting with simple for loops to iterate through lists. Practice with concrete examples like printing numbers 1 to 10. Then progress to while loops. Draw diagrams to visualize how loops work. Practice is essential for understanding.',
                'type': 'contextual',
                'quality': 'high'
            },
        ]

        return self._format_conversations(contextual)

    def _create_factual_data(self) -> List[Dict]:
        """Create factual knowledge examples"""

        factual = [
            {
                'input': 'What is photosynthesis?',
                'output': 'Photosynthesis is the process by which plants convert light energy into chemical energy. Plants use sunlight, water, and carbon dioxide to produce glucose (sugar) and oxygen. This process occurs primarily in the chloroplasts of plant cells.',
                'type': 'factual',
                'quality': 'high'
            },
            {
                'input': 'Who invented the telephone?',
                'output': 'Alexander Graham Bell is credited with inventing the telephone in 1876. He was awarded the first patent for the electric telephone.',
                'type': 'factual',
                'quality': 'high'
            },
            {
                'input': 'What is the speed of light?',
                'output': 'The speed of light in a vacuum is approximately 299,792,458 meters per second, commonly rounded to 300,000 kilometers per second or about 186,000 miles per second. It is denoted by the constant "c" in physics.',
                'type': 'factual',
                'quality': 'high'
            },
        ]

        return self._format_conversations(factual)

    def _create_reasoning_data(self) -> List[Dict]:
        """Create reasoning and logic examples"""

        reasoning = [
            {
                'input': 'If all cats are mammals, and all mammals are animals, what can we conclude about cats?',
                'output': 'We can conclude that all cats are animals. This follows from logical transitivity: if A implies B, and B implies C, then A implies C.',
                'type': 'reasoning',
                'quality': 'high'
            },
            {
                'input': 'A bag contains 5 red balls and 3 blue balls. What is the probability of picking a red ball?',
                'output': 'The probability of picking a red ball is 5/8 or 0.625 (62.5%). This is calculated by dividing the number of red balls (5) by the total number of balls (8).',
                'type': 'reasoning',
                'quality': 'high'
            },
            {
                'input': 'If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?',
                'output': 'It would still take 5 minutes. Each machine makes one widget in 5 minutes. So 100 machines working in parallel would make 100 widgets in 5 minutes.',
                'type': 'reasoning',
                'quality': 'high'
            },
        ]

        return self._format_conversations(reasoning)

    def _create_hindi_conversations(self) -> List[Dict]:
        """Create Hindi language conversation pairs"""

        hindi = [
            {
                'input': 'नमस्ते! आप कैसे हैं?',
                'output': 'नमस्ते! मैं ठीक हूं, धन्यवाद। मैं आपकी कैसे मदद कर सकता हूं?',
                'type': 'hindi_greeting',
                'quality': 'high'
            },
            {
                'input': 'मशीन लर्निंग क्या है?',
                'output': 'मशीन लर्निंग कृत्रिम बुद्धिमत्ता की एक शाखा है जहां कंप्यूटर डेटा से पैटर्न सीखते हैं। यह सिस्टम अनुभव और डेटा विश्लेषण के माध्यम से कार्य में अपने प्रदर्शन में सुधार करता है।',
                'type': 'hindi_explanation',
                'quality': 'high'
            },
            {
                'input': 'क्या आप हिंदी में बात कर सकते हैं?',
                'output': 'हां, मैं हिंदी में बात कर सकता हूं। मुझे आपकी मदद करने में खुशी होगी।',
                'type': 'hindi_capability',
                'quality': 'high'
            },
            {
                'input': 'मुझे प्रोग्रामिंग सीखनी है। कहां से शुरू करूं?',
                'output': 'प्रोग्रामिंग सीखने के लिए, पहले Python जैसी शुरुआती-अनुकूल भाषा चुनें। छोटे प्रोजेक्ट बनाकर नियमित रूप से अभ्यास करें। ट्यूटोरियल देखें और कोडिंग चुनौतियों को हल करें। निरंतरता महत्वपूर्ण है।',
                'type': 'hindi_instruction',
                'quality': 'high'
            },
            {
                'input': 'धन्यवाद!',
                'output': 'आपका स्वागत है! यदि आपके कोई और प्रश्न हैं, तो पूछने में संकोच न करें।',
                'type': 'hindi_acknowledgment',
                'quality': 'high'
            },
        ]

        return self._format_conversations(hindi)

    def _create_anime_conversations(self) -> List[Dict]:
        """Create anime domain conversations"""

        anime = [
            {
                'input': 'What is anime?',
                'output': 'Anime refers to Japanese animated productions featuring hand-drawn or computer animation. It encompasses various genres and styles, ranging from action and adventure to romance and slice-of-life. Anime has gained worldwide popularity.',
                'type': 'anime_explanation',
                'quality': 'high'
            },
            {
                'input': 'Can you recommend a good action anime?',
                'output': 'For action anime, I recommend Attack on Titan (Shingeki no Kyojin). It features intense battles, complex characters, and a gripping storyline about humanity fighting giant humanoid creatures. The animation quality and plot development are exceptional.',
                'type': 'anime_recommendation',
                'quality': 'high'
            },
            {
                'input': 'मुझे एनीमे के बारे में बताएं',
                'output': 'एनीमे जापानी एनिमेशन को कहते हैं। यह विभिन्न शैलियों में आता है जैसे एक्शन, रोमांस, फंतासी, और साइंस फिक्शन। एनीमे दुनिया भर में बहुत लोकप्रिय है।',
                'type': 'anime_hindi',
                'quality': 'high'
            },
            {
                'input': 'What are the most popular anime genres?',
                'output': 'Popular anime genres include Shonen (action/adventure for young males), Shoujo (romance for young females), Seinen (mature themes), Isekai (transported to another world), Slice of Life, Mecha (robots), and Supernatural. Each genre offers unique storytelling styles.',
                'type': 'anime_knowledge',
                'quality': 'high'
            },
            {
                'input': 'Tell me about Studio Ghibli.',
                'output': 'Studio Ghibli is a renowned Japanese animation studio founded by Hayao Miyazaki and Isao Takahata. Known for beautiful hand-drawn animation and heartfelt storytelling, their films like Spirited Away, My Neighbor Totoro, and Princess Mononoke have won international acclaim.',
                'type': 'anime_knowledge',
                'quality': 'high'
            },
        ]

        return self._format_conversations(anime)

    def _format_conversations(self, conversations: List[Dict]) -> List[Dict]:
        """Format conversations with proper structure"""

        formatted = []

        for conv in conversations:
            formatted_conv = {
                'text': f"User: {conv['input']}\nAssistant: {conv['output']}",
                'input': conv['input'],
                'output': conv['output'],
                'type': conv.get('type', 'general'),
                'quality': conv.get('quality', 'high'),
                'verified': True
            }
            formatted.append(formatted_conv)

        return formatted

    def _validate_dataset(self, dataset: List[Dict]) -> List[Dict]:
        """Validate entire dataset for quality"""

        validated = []

        for item in dataset:
            if self._validate_item(item):
                validated.append(item)

        return validated

    def _validate_item(self, item: Dict) -> bool:
        """Validate single item"""

        # Check required fields
        if 'text' not in item:
            return False

        text = item['text']

        # Check length
        if len(text.split()) < 5 or len(text.split()) > 500:
            return False

        # Check quality marker
        if item.get('quality') == 'low':
            return False

        return True

    def save_dataset(self, dataset: List[Dict], output_path: str):
        """Save dataset to file"""

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        print(f"\n✓ Dataset saved to: {output_path}")
        print(f"✓ Total examples: {len(dataset)}")

    def load_dataset(self, input_path: str) -> List[Dict]:
        """Load dataset from file"""

        with open(input_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        print(f"\n✓ Loaded dataset from: {input_path}")
        print(f"✓ Total examples: {len(dataset)}")

        return dataset


def create_comprehensive_dataset() -> List[Dict]:
    """Create comprehensive high-quality dataset"""

    creator = ConversationalDatasetCreator()
    dataset = creator.create_conversational_dataset()

    return dataset
