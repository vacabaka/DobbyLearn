from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from contextlib import asynccontextmanager
import logging
import os
from dotenv import load_dotenv
import aiohttp

from database import DatabaseManager
from ai_service import DobbyAIService

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

db = DatabaseManager()
ai_service = DobbyAIService(os.getenv('FIREWORKS_API_KEY', ''))

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events"""
    # Startup
    await db.init_database()
    logger.info("🚀 DobbyLearn API started")
    yield
    # Shutdown
    await db.close()

app = FastAPI(
    title="DobbyLearn API",
    description="API для изучения слов",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === МОДЕЛИ PYDANTIC ===

class GroupResponse(BaseModel):
    """Ответ с группой слов"""
    id: int
    name: str
    description: Optional[str]
    icon: str
    color: str
    total_words: int
    learned_words: int

class WordResponse(BaseModel):
    """Ответ со словом"""
    id: int
    word: str
    translation: Optional[str]
    knowledge_level: int
    next_review: datetime
    learned: bool

class ReviewRequest(BaseModel):
    """Запрос на обновление прогресса"""
    word_id: int
    difficulty: str  # 'again', 'hard', 'good', 'easy'

# === ЭНДПОИНТЫ ===

@app.get("/")
async def root():
    """Главная страница - отдать index.html"""
    return FileResponse("index.html")

@app.get("/api/health")
async def health_check():
    """Проверка здоровья API"""
    return {"status": "ok", "message": "DobbyLearn API is running"}

@app.get("/api/groups/{telegram_id}")
async def get_groups(telegram_id: int):
    """Получить все группы пользователя"""
    try:
        user = await db.get_user_by_telegram_id(telegram_id)
        if not user:
            raise HTTPException(status_code=404, detail="Пользователь не найден")
        
        groups = await db.get_user_groups(user.id)
        
        return [
            {
                "id": g.id,
                "name": g.name,
                "description": g.description,
                "icon": g.icon,
                "color": g.color,
                "total_words": g.total_words,
                "learned_words": g.learned_words,
                "native_language": g.native_language,
                "target_language": g.target_language
            }
            for g in groups
        ]
    except Exception as e:
        logger.error(f"Ошибка получения групп: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/words/{telegram_id}")
async def get_words_for_review(telegram_id: int, group_id: Optional[int] = None, limit: int = 20):
    """
    Получить слова для повторения 
    - telegram_id: ID пользователя в Telegram
    - group_id: (опционально) ID группы для фильтрации
    - limit: количество слов (по умолчанию 20)
    """
    try:
        user = await db.get_user_by_telegram_id(telegram_id)
        if not user:
            raise HTTPException(status_code=404, detail="Пользователь не найден")
        
        words = await db.get_words_for_review(user.id, group_id, limit)
        
        # Получить группы для добавления информации о языках
        groups = await db.get_user_groups(user.id)
        group_languages = {g.id: (g.target_language, g.native_language) for g in groups}
        
        return [
            {
                "id": w.id,
                "word": w.word,
                "translation": w.translation,
                "knowledge_level": w.knowledge_level,
                "next_review": w.next_review.isoformat() if w.next_review else None,
                "learned": w.learned,
                "review_count": w.review_count,
                "correct_count": w.correct_count,
                "wrong_count": w.wrong_count,
                "group_id": w.group_id,
                "word_language": group_languages.get(w.group_id, ("en", "ru"))[0],  # target_language группы
                "translation_language": group_languages.get(w.group_id, ("en", "ru"))[1]  # native_language
            }
            for w in words
        ]
    except Exception as e:
        logger.error(f"Ошибка получения слов: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/review")
async def update_review(review: ReviewRequest):
    """
    Обновить прогресс изучения 
    
    Параметры:
    - word_id: ID слова
    - difficulty: уровень сложности ('again', 'hard', 'good', 'easy')
    
    """
    try:
        if review.difficulty not in ['again', 'hard', 'good', 'easy']:
            raise HTTPException(
                status_code=400, 
                detail="difficulty должен быть: again, hard, good, или easy"
            )
        
        success = await db.update_word_progress_anki(review.word_id, review.difficulty)
        
        if not success:
            raise HTTPException(status_code=404, detail="Слово не найдено")
        
        return {
            "success": True,
            "message": f"Прогресс обновлен: {review.difficulty}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка обновления прогресса: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/words/group/{group_id}")
async def get_group_words(group_id: int):
    """Получить все слова в группе"""
    try:
        words = await db.get_group_words(group_id)
        
        return [
            {
                "id": w.id,
                "word": w.word,
                "translation": w.translation,
                "learned": w.learned,
                "knowledge_level": w.knowledge_level,
                "review_count": w.review_count
            }
            for w in words
        ]
    except Exception as e:
        logger.error(f"Ошибка получения слов группы: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class AIExampleRequest(BaseModel):
    """Запрос на генерацию AI примера"""
    word: str
    native_language: str = "ru"
    target_language: str = "en"
    style: str = "casual"  # casual, formal, slang

@app.post("/api/ai/example")
async def generate_ai_example(request: AIExampleRequest):
    """
    Сгенерировать AI пример для слова
    
    Параметры:
    - word: слово для которого нужен пример
    - native_language: язык перевода (ru, en, es, fr, de, tr)
    - target_language: язык изучаемого слова
    - style: стиль примера (casual, formal, slang)
    
    Возвращает:
    - example: пример использования слова
    - translation: перевод примера на базовый язык
    """
    try:
        result = await ai_service.generate_single_example(
            word=request.word,
            style=request.style,
            translation="",  
            language=request.target_language  # Язык слова (испанский, английский, и т.д.)
        )
        
        if not result:
            raise HTTPException(status_code=500, detail="AI не смог сгенерировать пример")
        
        return {
            "success": True,
            "example": result.get('example_text', ''),
            "translation": result.get('example_translation', ''),  
            "context": result.get('context', ''),
            "word": request.word
        }
    except Exception as e:
        logger.error(f"Ошибка генерации примера: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/translate")
async def translate_word(word: str, native_language: str = "ru", target_language: str = "en"):
    """
    Перевести слово
    
    Параметры:
    - word: слово для перевода
    - native_language: на какой язык переводить
    - target_language: с какого языка переводить
    
    Возвращает:
    - translation: перевод слова
    - word: исходное слово
    """
    try:
        result = await ai_service.translate_word(
            word=word,
            target_language=target_language,
            native_language=native_language
        )
        
        return {
            "success": True,
            "word": word,
            "translation": result.get('translation')
        }
    except Exception as e:
        logger.error(f"Ошибка перевода: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === LEGACY ENDPOINTS (для совместимости с фронтендом) ===

@app.post("/api/generate-example")
async def generate_example_legacy(request: AIExampleRequest):
    """
    Legacy эндпоинт для генерации примеров (для совместимости с index.html)
    Перенаправляет на /api/ai/example
    """
    return await generate_ai_example(request)

def generate_simple_pronunciation(word: str) -> str:
    """
    Генерирует упрощенную IPA транскрипцию на основе базовых правил английского
    """
    word = word.lower()
    result = []
    
    # Базовая таблица соответствий
    vowel_map = {
        'a': 'æ', 'e': 'ɛ', 'i': 'ɪ', 'o': 'ɑ', 'u': 'ʌ',
        'ee': 'iː', 'ea': 'iː', 'oo': 'uː', 'ou': 'aʊ',
        'ow': 'oʊ', 'ay': 'eɪ', 'ai': 'eɪ', 'oy': 'ɔɪ',
        'igh': 'aɪ', 'ie': 'aɪ'
    }
    
    consonant_map = {
        'ch': 'tʃ', 'sh': 'ʃ', 'th': 'θ', 'ph': 'f',
        'ng': 'ŋ', 'qu': 'kw', 'ck': 'k'
    }
    
    i = 0
    while i < len(word):
        # Проверяем двойные буквы
        if i < len(word) - 1:
            two_chars = word[i:i+2]
            if two_chars in consonant_map:
                result.append(consonant_map[two_chars])
                i += 2
                continue
            if two_chars in vowel_map:
                result.append(vowel_map[two_chars])
                i += 2
                continue
        
        # Проверяем тройные буквы (igh)
        if i < len(word) - 2:
            three_chars = word[i:i+3]
            if three_chars in vowel_map:
                result.append(vowel_map[three_chars])
                i += 3
                continue
        
        # Одиночные буквы
        char = word[i]
        if char in vowel_map:
            result.append(vowel_map[char])
        elif char == 'y' and i > 0:
            result.append('i')
        else:
            result.append(char)
        i += 1
    
    return f"/{''.join(result)}/"

@app.post("/api/generate-pronunciation")
async def generate_pronunciation(data: dict):
    """
    Сгенерировать произношение для слова (IPA)
    
    Принимает:
    - word_id: ID слова в базе (фронтенд отправляет это)
    или
    - word: текст слова напрямую
    
    Возвращает:
    - pronunciation: IPA транскрипция
    """
    logger.info(f"📢 Pronunciation request: {data}")
    
    word = None
    
    # Если пришел word_id - получить слово из БД
    word_id = data.get('word_id') or data.get('wordId') or data.get('id')
    if word_id:
        logger.info(f"🔍 Looking up word by ID: {word_id}")
        try:
            # Получить слово из базы данных
            from sqlalchemy import select
            from models import Word
            
            async with db.get_session() as session:
                result = await session.execute(
                    select(Word.word).where(Word.id == word_id)
                )
                row = result.first()
                if row:
                    word = row[0]
                    logger.info(f"✅ Found word in DB: {word}")
        except Exception as e:
            logger.error(f"Error getting word from DB: {e}")
    
    # Если не нашли по ID, попробовать извлечь напрямую
    if not word:
        word = data.get('word') or data.get('text') or data.get('term') or data.get('value')
    
    if not word:
        logger.warning(f"⚠️ Could not find word: {data}")
        return {
            "success": False,
            "error": "Word not found",
            "pronunciation": ""
        }
    
    logger.info(f"🔤 Generating pronunciation for: {word}")
    
    # словарь частых слов с IPA
    common_ipa = {
        "heavy": "ˈhɛvi",
        "hello": "həˈloʊ",
        "world": "wɜːrld",
        "example": "ɪɡˈzæmpəl",
        "learn": "lɜːrn",
        "study": "ˈstʌdi",
        "book": "bʊk",
        "house": "haʊs",
        "cat": "kæt",
        "dog": "dɔːɡ",
        "water": "ˈwɔːtər",
        "food": "fuːd",
        "love": "lʌv",
        "friend": "frɛnd",
        "family": "ˈfæməli",
        "apple": "ˈæpəl",
        "coffee": "ˈkɔːfi",
        "computer": "kəmˈpjuːtər",
        "beautiful": "ˈbjuːtəfəl",
        "important": "ɪmˈpɔːrtənt",
        "time": "taɪm",
        "ocean": "ˈoʊʃən",
        "thought": "θɔːt",
        "future": "ˈfjuːtʃər",
        "rhythm": "ˈrɪðəm",
        "energy": "ˈɛnərdʒi",
        "create": "kriˈeɪt",
        "balance": "ˈbæləns",
        "whisper": "ˈwɪspər",
        "shadow": "ˈʃædoʊ",
        "school": "skuːl",
        "happy": "ˈhæpi",
        "travel": "ˈtrævəl",
        "strong": "strɔːŋ",
        "night": "naɪt",
        "new": "njuː",
        "try": "traɪ",
        "people": "ˈpiːpəl",
        "work": "wɜːrk",
        "good": "ɡʊd",
        "great": "ɡreɪt",
        "thing": "θɪŋ",
        "know": "noʊ",
        "think": "θɪŋk",
        "year": "jɪr",
        "day": "deɪ",
        "life": "laɪf",
        "hand": "hænd",
        "part": "pɑːrt"
    }
    
    # 1. Проверить в локальном словаре
    if word.lower() in common_ipa:
        pronunciation = f"/{common_ipa[word.lower()]}/"
        logger.info(f"✅ Found in local dictionary: {pronunciation}")
    else:
        # получить из FreeDictionaryAPI
        pronunciation = None
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word.lower()}"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data and len(data) > 0:
                            phonetics = data[0].get('phonetics', [])
                            for phonetic in phonetics:
                                ipa_text = phonetic.get('text', '')
                                if ipa_text and len(ipa_text) > 2:
                                    pronunciation = ipa_text.strip('/')
                                    pronunciation = f"/{pronunciation}/"
                                    logger.info(f"✅ Found via FreeDictionaryAPI: {pronunciation}")
                                    break
        except Exception as e:
            logger.warning(f"⚠️ FreeDictionaryAPI failed: {str(e)[:100]}")
        
        # 3. при ошибки генерируем базовую транскрипцию
        if not pronunciation:
            # Простые правила английской транскрипции для fallback
            pronunciation = generate_simple_pronunciation(word)
            logger.info(f"📝 Generated simple pronunciation: {pronunciation}")
    
    logger.info(f"✅ Final pronunciation result: {pronunciation}")
    
    return {
        "success": True,
        "word": word,
        "pronunciation": pronunciation
    }

@app.post("/api/update-progress")
async def update_progress(data: dict):
    """
    Обновить прогресс
    
    Принимает
    - word_id: ID слова
    - quality или difficulty: оценка знания
    
     кнопки:
    - Again (0) = снова
    - Hard (1) = сложно  
    - Good (2) = хорошо
    - Easy (3) = легко
    """
    try:
        logger.info(f"📊 Update progress request: {data}")
        
        word_id = data.get('word_id') or data.get('wordId') or data.get('id')
        
        if not word_id:
            logger.error("❌ No word_id in request")
            raise HTTPException(status_code=400, detail="word_id is required")
        
        logger.info(f"🆔 Word ID: {word_id}")
        
        # Конвертировать quality (0-3) в difficulty (again/hard/good/easy)
        quality = data.get('quality')
        difficulty = data.get('difficulty')
        
        logger.info(f"📈 Quality: {quality}, Difficulty: {difficulty}")
        
        if quality is not None:
            # Маппинг quality -> difficulty
            quality_map = {
                0: 'again',
                1: 'hard',
                2: 'good',
                3: 'easy'
            }
            difficulty = quality_map.get(int(quality), 'good')
            logger.info(f"🔄 Mapped quality {quality} -> difficulty {difficulty}")
        
        if not difficulty or difficulty not in ['again', 'hard', 'good', 'easy']:
            difficulty = 'good'  # Default
            logger.warning(f"⚠️ Using default difficulty: {difficulty}")
        
        logger.info(f"💾 Updating DB: word_id={word_id}, difficulty={difficulty}")
        
        # Обновить прогресс в БД
        success = await db.update_word_progress_anki(word_id, difficulty)
        
        if not success:
            logger.error(f"❌ DB update failed for word_id={word_id}")
            raise HTTPException(status_code=404, detail="Word not found")
        
        logger.info(f"✅ Progress updated successfully: {difficulty}")
        
        # Получить user_id из слова для поиска следующего
        from sqlalchemy import select
        from models import Word
        
        async with db.get_session() as session:
            result = await session.execute(
                select(Word.user_id, Word.group_id).where(Word.id == word_id)
            )
            row = result.first()
            if row:
                user_id = row[0]
                group_id = row[1]
                
                # Получить следующее слово для повторения (исключить текущее!)
                next_words = await db.get_words_for_review(user_id, group_id, limit=5)
                
                # Исключить только что обновленное слово
                logger.info(f"🔍 Before filter: {len(next_words)} words, excluding word_id={word_id}")
                next_words = [w for w in next_words if w.id != word_id]
                logger.info(f"🔍 After filter: {len(next_words)} words remaining")
                
                if next_words:
                    next_words = [next_words[0]]  # Взять первое из оставшихся
                
                response = {
                    "success": True,
                    "word_id": word_id,
                    "difficulty": difficulty,
                    "message": f"Progress updated: {difficulty}",
                    "next_word": None
                }
                
                if next_words:
                    next_word = next_words[0]
                    
                    # ИСПРАВЛЕНИЕ: Получить языки группы для next_word
                    from sqlalchemy import select
                    from models import WordGroup
                    
                    async with db.get_session() as session:
                        result = await session.execute(
                            select(WordGroup.target_language, WordGroup.native_language)
                            .where(WordGroup.id == group_id)
                        )
                        group_langs = result.first()
                        word_lang = group_langs[0] if group_langs else 'en'
                        trans_lang = group_langs[1] if group_langs else 'ru'
                    
                    response["next_word"] = {
                        "id": next_word.id,
                        "word": next_word.word,
                        "translation": next_word.translation,
                        "knowledge_level": next_word.knowledge_level,
                        "learned": next_word.learned,
                        "next_review": next_word.next_review.isoformat() if next_word.next_review else None,
                        "word_language": word_lang,  # Язык слова (target_language группы)
                        "translation_language": trans_lang  # Язык перевода (native_language)
                    }
                    logger.info(f"📤 Sending next word: {next_word.word} (id={next_word.id}, level={next_word.knowledge_level}, next={next_word.next_review}, lang={word_lang})")
                else:
                    logger.info("✨ No more words to review!")
                
                return response
        
        # Fallback 
        return {
            "success": True,
            "word_id": word_id,
            "difficulty": difficulty,
            "message": f"Progress updated: {difficulty}",
            "next_word": None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"💥 Error updating progress: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Статичка 
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
