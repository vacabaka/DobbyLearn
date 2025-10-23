from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()

class User(Base):
    """Пользователь Telegram"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    telegram_id = Column(Integer, unique=True, index=True, nullable=False)
    username = Column(String(255), nullable=True)
    first_name = Column(String(255), nullable=True)
    last_name = Column(String(255), nullable=True)
    
    # Настройки языка
    native_language = Column(String(10), default="ru")  # Базовый язык (для переводов)
    language_setup_complete = Column(Boolean, default=False)  # Завершена ли настройка языка
    
    # Временные метки
    created_at = Column(DateTime, default=func.now())
    last_active = Column(DateTime, default=func.now())
    
    # Relationships
    word_groups = relationship("WordGroup", back_populates="user", cascade="all, delete-orphan")
    words = relationship("Word", back_populates="user", cascade="all, delete-orphan")

class WordGroup(Base):
    """Группа слов для организации"""
    __tablename__ = "word_groups"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    icon = Column(String(10), default="📚")
    color = Column(String(7), default="#667eea")
    created_at = Column(DateTime, default=func.now())
    is_active = Column(Boolean, default=True)
    
    # Языковые настройки группы
    native_language = Column(String(10), default="ru")
    target_language = Column(String(10), default="en")
    
    # Статистика
    total_words = Column(Integer, default=0)
    learned_words = Column(Integer, default=0)
    
    # Relationships
    user = relationship("User", back_populates="word_groups")
    words = relationship("Word", back_populates="group", cascade="all, delete-orphan")

class Word(Base):
    """Слово с системой повторений"""
    __tablename__ = "words"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    group_id = Column(Integer, ForeignKey("word_groups.id"), nullable=True)
    
    # Данные слова
    word = Column(String(255), nullable=False, index=True)
    translation = Column(Text, nullable=True)
    
    # Система повторений
    knowledge_level = Column(Integer, default=0)  # 0=новое, 1-4+ уровень знания
    next_review = Column(DateTime, default=func.now())  # Когда показать снова
    last_reviewed = Column(DateTime, nullable=True)  # Последний просмотр
    review_count = Column(Integer, default=0)  # Сколько раз повторяли
    correct_count = Column(Integer, default=0)  # Сколько раз правильно
    wrong_count = Column(Integer, default=0)  # Сколько раз неправильно
    
    # Флаг изучения
    learned = Column(Boolean, default=False)
    
    # Метаданные
    created_at = Column(DateTime, default=func.now())
    added_via = Column(String(20), default="chat")  # chat, manual
    
    # Relationships
    user = relationship("User", back_populates="words")
    group = relationship("WordGroup", back_populates="words")
