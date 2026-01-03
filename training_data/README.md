# Training Datasets

This directory contains manually curated, high-quality training datasets for the Analazy chatbot. Each dataset focuses on specific aspects of conversational AI and anime knowledge.

## Dataset Overview

### 1. conversational_training.csv (20 entries)
**Purpose**: Natural conversation patterns and greetings
**Coverage**:
- Bilingual greetings (Hindi/English)
- Anime recommendations with context
- Language capability demonstrations
- General anime discussions

**Key Features**:
- Mixed Hindi-English responses
- Contextual understanding
- Personalized recommendations

### 2. anime_knowledge.csv (20 entries)
**Purpose**: Factual anime information database
**Coverage**:
- Character information and motivations
- Power systems (Devil Fruits, Nen, Breathing Styles, etc.)
- Series creators and studios
- Plot mechanics and endings
- Technical terminology

**Difficulty Levels**: Easy, Medium, Hard

### 3. reasoning_responses.csv (20 entries)
**Purpose**: Analytical and comparative discussions
**Coverage**:
- Thematic analysis (friendship, storytelling)
- Comparative analysis (series comparisons)
- Cultural analysis (anime industry, trends)
- Genre analysis (slice of life, isekai)
- Quality assessment

**Reasoning Types**:
- Emotional analysis
- Storytelling analysis
- Trend analysis
- Audience analysis

### 4. hindi_english_mix.csv (20 entries)
**Purpose**: Natural Hinglish conversation style
**Coverage**:
- Casual Hinglish chat patterns
- Code-switching fluency
- Colloquial expressions
- Cultural references
- Review and critique style

**Language Styles**:
- hinglish_casual
- hinglish_conversational
- hinglish_analytical
- hinglish_critical

### 5. instruction_following.csv (20 entries)
**Purpose**: Structured task responses
**Coverage**:
- List-based recommendations
- Concise explanations
- Comparison tasks
- Definition requests
- Balanced opinions

**Sample Instructions**:
- "Give me 3 recommendations"
- "Explain in one sentence"
- "List top 5..."
- "Compare X vs Y"

### 6. advanced_queries.csv (19 entries)
**Purpose**: Expert-level philosophical and technical discussions
**Coverage**:
- Philosophical themes (Evangelion, Monster)
- Power system design comparisons
- Cultural significance analysis
- Director auteur theory
- Genre deconstruction
- Animation technique analysis

**Expertise Level**: Advanced to Expert

## Dataset Quality Standards

### Manual Curation
- Every entry hand-written by human expert
- No auto-generated or scraped content
- Verified facts and accurate information

### Natural Language
- Conversational tone, not robotic
- Context-aware responses
- Cultural sensitivity

### Multilingual Quality
- Proper Hindi-English code-switching
- Natural Hinglish patterns
- Grammatically correct in both languages

### Low Hallucination Design
- Fact-based responses only
- "I don't know" when uncertain
- Sources and reasoning provided
- No fabricated information

## Training Recommendations

### Best Practices
1. **Balanced Loading**: Use all datasets in training for diverse capabilities
2. **Quality Metrics**: Monitor hallucination rate and response naturalness
3. **Validation Split**: Hold out 15-20% for testing
4. **Augmentation**: Can expand by paraphrasing while maintaining accuracy
5. **Iterative Improvement**: Add new entries based on failure cases

### Target Metrics
- **Response Naturalness**: >95%
- **Factual Accuracy**: >98%
- **Hallucination Rate**: <5%
- **Language Mixing Fluency**: >90%
- **Instruction Following**: >95%

## Dataset Statistics

| Dataset | Entries | Avg Response Length | Languages | Complexity |
|---------|---------|---------------------|-----------|------------|
| Conversational | 20 | Medium | Mixed | Basic-Medium |
| Anime Knowledge | 20 | Medium | Mixed | Easy-Hard |
| Reasoning | 20 | Long | Mixed | Medium-High |
| Hindi-English Mix | 20 | Long | Hinglish | Medium |
| Instruction Following | 20 | Medium | English/Hindi | Basic-Medium |
| Advanced Queries | 19 | Very Long | English | Advanced-Expert |

**Total**: 119 high-quality training examples

## Extending Datasets

When adding new entries, maintain:
- Factual accuracy (verify all claims)
- Natural conversational tone
- Diverse question types
- Appropriate difficulty distribution
- Cultural sensitivity
- Proper CSV formatting
