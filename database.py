import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select, update, delete, func
from sqlalchemy.orm import sessionmaker
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from models import Base, User, WordGroup, Word

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, database_url: str = "sqlite+aiosqlite:///db/dobbylearn.db"):
        self.database_url = database_url
        self.engine = create_async_engine(database_url, echo=False)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_database(self):
        """Инициализация таблиц базы данных"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("База данных инициализирована")
    
    async def init_db(self):
        return await self.init_database()

    def get_session(self) -> AsyncSession:
        """Получить сессию базы данных"""
        return self.async_session()

    #УПРАВЛЕНИЕ ПОЛЬЗОВАТЕЛЯМИ
    
    async def create_or_get_user(self, telegram_id: int, username: str = None, 
                               first_name: str = None, last_name: str = None) -> User:
        """Создать или получить пользователя"""
        return await self.get_or_create_user(telegram_id, username, first_name, last_name)
    
    async def get_or_create_user(self, telegram_id: int, username: str = None, 
                               first_name: str = None, last_name: str = None) -> User:
        """Получить существующего пользователя или создать нового"""
        async with self.get_session() as session:
            result = await session.execute(
                select(User).where(User.telegram_id == telegram_id)
            )
            user = result.scalar_one_or_none()
            
            if user:
                # Обновить данные
                user.last_active = func.now()
                if username:
                    user.username = username
                if first_name:
                    user.first_name = first_name
                if last_name:
                    user.last_name = last_name
                await session.commit()
                return user
            
            # Создать нового
            user = User(
                telegram_id=telegram_id,
                username=username,
                first_name=first_name,
                last_name=last_name
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            
            # Дефолтная группа теперь создается после выбора языков в боте
            
            logger.info(f"Создан пользователь: {telegram_id}")
            return user

    async def get_user_by_telegram_id(self, telegram_id: int) -> Optional[User]:
        """Получить пользователя по Telegram ID"""
        async with self.get_session() as session:
            result = await session.execute(
                select(User).where(User.telegram_id == telegram_id)
            )
            return result.scalar_one_or_none()

    # УПРАВЛЕНИЕ ГРУППАМИ
    
    async def create_default_group(self, user_id: int, native_language: str = "ru", target_language: str = "en"):
        """Создать группу по умолчанию"""
        async with self.get_session() as session:
            default_group = WordGroup(
                user_id=user_id,
                name="Мои слова",
                description="Основная группа слов",
                icon="📚",
                color="#667eea",
                native_language=native_language,
                target_language=target_language
            )
            session.add(default_group)
            await session.commit()

    async def create_word_group(self, user_id: int, name: str, description: str = "", 
                              icon: str = "📚", color: str = "#667eea",
                              native_language: str = "ru", target_language: str = "en") -> Optional[WordGroup]:
        """Создать новую группу слов"""
        async with self.get_session() as session:
            group = WordGroup(
                user_id=user_id,
                name=name,
                description=description,
                icon=icon,
                color=color,
                native_language=native_language,
                target_language=target_language
            )
            session.add(group)
            await session.commit()
            await session.refresh(group)
            return group

    async def get_user_groups(self, user_id: int) -> List[WordGroup]:
        """Получить все активные группы пользователя"""
        async with self.get_session() as session:
            result = await session.execute(
                select(WordGroup)
                .where(WordGroup.user_id == user_id, WordGroup.is_active == True)
                .order_by(WordGroup.created_at)
            )
            return result.scalars().all()

    async def delete_group(self, group_id: int, user_id: int) -> bool:
        """Удалить группу"""
        async with self.get_session() as session:
            result = await session.execute(
                select(WordGroup).where(
                    WordGroup.id == group_id,
                    WordGroup.user_id == user_id
                )
            )
            group = result.scalar_one_or_none()
            
            if group:
                group.is_active = False
                await session.commit()
                return True
            return False

    async def update_group_stats(self, group_id: int):
        """Обновить статистику группы"""
        async with self.get_session() as session:
            # Подсчет слов
            total_result = await session.execute(
                select(func.count(Word.id)).where(Word.group_id == group_id)
            )
            total_words = total_result.scalar() or 0
            
            # Подсчет изученных
            learned_result = await session.execute(
                select(func.count(Word.id)).where(
                    Word.group_id == group_id,
                    Word.learned == True
                )
            )
            learned_words = learned_result.scalar() or 0
            
            # Обновление
            await session.execute(
                update(WordGroup).where(WordGroup.id == group_id).values(
                    total_words=total_words,
                    learned_words=learned_words
                )
            )
            await session.commit()

    # управление словами 
    
    async def add_word(self, user_id: int, word_data: Dict[str, Any], group_id: int = None) -> Word:
        """Добавить новое слово"""
        async with self.get_session() as session:
            # Получить группу по умолчанию если не указана
            if not group_id:
                result = await session.execute(
                    select(WordGroup).where(
                        WordGroup.user_id == user_id,
                        WordGroup.is_active == True
                    ).order_by(WordGroup.created_at).limit(1)
                )
                default_group = result.scalar_one_or_none()
                if default_group:
                    group_id = default_group.id

            word = Word(
                user_id=user_id,
                group_id=group_id,
                word=word_data.get('word', '').lower().strip(),
                translation=word_data.get('translation'),
                added_via=word_data.get('added_via', 'chat')
            )
            
            session.add(word)
            await session.commit()
            await session.refresh(word)
            
            # Обновить статистику группы
            if group_id:
                await self.update_group_stats(group_id)
            
            return word

    async def get_words_for_review(self, user_id: int, group_id: int = None, limit: int = 20) -> List[Word]:
        """Получить слова для повторения"""
        async with self.get_session() as session:
            current_time = datetime.now()  # Использовать локальное время как и в update
            
            # Построить запрос с учетом NULL значений
            from sqlalchemy import or_
            
            query = select(Word).where(
                Word.user_id == user_id,
                or_(
                    Word.next_review.is_(None),  # Новые слова без даты
                    Word.next_review <= current_time  # Или пришло время повторения
                )
            )
            
            if group_id:
                query = query.where(Word.group_id == group_id)
            
            # Приоритет: новые слова > сложные > по времени
            query = query.order_by(
                Word.knowledge_level.asc(),  # Новые первыми
                Word.next_review.asc().nulls_first()  # Новые (NULL) первыми, потом старые
            ).limit(limit)
            
            result = await session.execute(query)
            words = result.scalars().all()
            
            logger.info(f"📋 Words for review: found {len(words)} words (user={user_id}, group={group_id})")
            for w in words[:3]:  # Логируем первые 3
                logger.info(f"  - {w.word} (id={w.id}, level={w.knowledge_level}, next={w.next_review})")
            
            return words

    async def get_user_words(self, user_id: int, group_id: int = None) -> List[Word]:
        """Получить все слова пользователя"""
        async with self.get_session() as session:
            query = select(Word).where(Word.user_id == user_id)
            
            if group_id:
                query = query.where(Word.group_id == group_id)
            
            query = query.order_by(Word.created_at.desc())
            
            result = await session.execute(query)
            return result.scalars().all()

    async def get_group_words(self, group_id: int) -> List[Word]:
        """Получить все слова в группе"""
        async with self.get_session() as session:
            result = await session.execute(
                select(Word).where(Word.group_id == group_id)
            )
            return result.scalars().all()

    # система повторений 
    
    async def update_word_progress_anki(self, word_id: int, difficulty_level: str) -> bool:
        """        
        Args:
            word_id: ID слова
            difficulty_level: 'again', 'hard', 'good', 'easy'
        
        короткие интервалы для новых слов:
        - Level 0 (учу сейчас): 1м, 10м, 1ч, 4ч (короткие интервалы!)
        - Level 1 (выучил недавно): 10м, 4ч, 1д, 3д
        - Level 2: 1д, 3д, 7д, 14д
        - Level 3: 1д, 7д, 14д, 30д
        - Level 4+: 1д, 14д, 30д, 60д
        
        Интервалы в минутах
        """
        INTERVALS_MINUTES = {
            # Level 0: короткие интервалы для новых слов
            0: {'again': 1, 'hard': 10, 'good': 60, 'easy': 240},  # 1м, 10м, 1ч, 4ч
            # Level 1: переход к дням
            1: {'again': 10, 'hard': 240, 'good': 1440, 'easy': 4320},  # 10м, 4ч, 1д, 3д
            # Level 2+: длинные интервалы 
            2: {'again': 1440, 'hard': 4320, 'good': 10080, 'easy': 20160},  # 1д, 3д, 7д, 14д
            3: {'again': 1440, 'hard': 10080, 'good': 20160, 'easy': 43200},  # 1д, 7д, 14д, 30д
            4: {'again': 1440, 'hard': 20160, 'good': 43200, 'easy': 86400},  # 1д, 14д, 30д, 60д
        }
        
        async with self.get_session() as session:
            try:
                result = await session.execute(select(Word).where(Word.id == word_id))
                word = result.scalar_one_or_none()
                
                if not word:
                    return False
                
                current_level = min(word.knowledge_level or 0, 4)
                interval_minutes = INTERVALS_MINUTES[current_level][difficulty_level]
                
                # Вычислить следующую дату повторения
                next_review = datetime.now() + timedelta(minutes=interval_minutes)
                
                # Обновить уровень знания
                if difficulty_level == 'again':
                    new_level = max(0, current_level - 1)
                elif difficulty_level == 'hard':
                    new_level = current_level
                elif difficulty_level == 'good':
                    new_level = current_level + 1
                elif difficulty_level == 'easy':
                    new_level = current_level + 2
                else:
                    new_level = current_level
                
                # Обновить слово
                word.knowledge_level = new_level
                word.next_review = next_review
                word.last_reviewed = datetime.now()
                word.review_count += 1
                
                if difficulty_level in ['good', 'easy']:
                    word.correct_count += 1
                    if new_level >= 3:
                        word.learned = True
                else:
                    word.wrong_count += 1
                    word.learned = False
                
                await session.commit()
                
                # Красивое логирование с правильными единицами
                if interval_minutes < 60:
                    interval_text = f"{interval_minutes} мин"
                elif interval_minutes < 1440:
                    interval_text = f"{interval_minutes // 60} ч"
                else:
                    interval_text = f"{interval_minutes // 1440} дн"
                
                logger.info(f"Слово {word_id} ({word.word}): уровень {current_level}→{new_level}, интервал {interval_text}, next_review={next_review.strftime('%H:%M:%S')}")
                return True
                
            except Exception as e:
                logger.error(f"Ошибка обновления прогресса: {e}")
                await session.rollback()
                return False

    async def delete_word(self, word_id: int) -> bool:
        """Удалить слово"""
        async with self.get_session() as session:
            try:
                await session.execute(delete(Word).where(Word.id == word_id))
                await session.commit()
                return True
            except Exception as e:
                await session.rollback()
                logger.error(f"Ошибка удаления слова: {e}")
                return False

    #  управление языками
    
    async def update_user_language_setup(self, user_id: int, native_language: str) -> bool:
        """Обновить базовый язык пользователя и пометить настройку завершенной"""
        async with self.get_session() as session:
            try:
                query = update(User).where(User.id == user_id).values(
                    native_language=native_language,
                    language_setup_complete=True
                )
                await session.execute(query)
                await session.commit()
                return True
            except Exception as e:
                logger.error(f"Ошибка обновления языка: {e}")
                await session.rollback()
                return False
    
    async def update_all_user_groups_language(self, user_id: int, native_language: str) -> bool:
        """Обновить native_language во всех группах пользователя"""
        async with self.get_session() as session:
            try:
                from models import WordGroup
                query = update(WordGroup).where(WordGroup.user_id == user_id).values(
                    native_language=native_language
                )
                result = await session.execute(query)
                await session.commit()
                logger.info(f"Обновлено {result.rowcount} групп для пользователя {user_id} на язык {native_language}")
                return True
            except Exception as e:
                logger.error(f"Ошибка обновления языка в группах: {e}")
                await session.rollback()
                return False
    
    async def get_user_language_setup_status(self, telegram_id: int) -> tuple:
        """Проверить завершена ли настройка языка"""
        async with self.get_session() as session:
            result = await session.execute(
                select(User.language_setup_complete, User.native_language)
                .where(User.telegram_id == telegram_id)
            )
            row = result.first()
            if row:
                return row[0] or False, row[1] or "ru"
            return False, "ru"
    
    async def get_language_tag(self, native_lang: str, target_lang: str) -> str:
        """Получить тег языка для отображения"""
        lang_codes = {
            "en": "ENG", "ru": "RUS", "es": "ESP", "fr": "FRA", 
            "de": "GER", "it": "ITA", "pt": "POR", "zh": "CHN", 
            "ja": "JPN", "tr": "TUR"
        }
        target_code = lang_codes.get(target_lang, target_lang.upper()[:3])
        return f"[{target_code}]"

    async def close(self):
        """Закрыть соединение"""
        await self.engine.dispose()
