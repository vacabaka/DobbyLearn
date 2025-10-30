import asyncio
import logging
import re
from typing import List
from dotenv import load_dotenv

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters
)

from database import DatabaseManager
from ai_service import DobbyAIService

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class DobbyLearnBot:
    def __init__(self, bot_token: str, fireworks_api_key: str, webapp_url: str):
        self.bot_token = bot_token
        self.webapp_url = webapp_url
        self.db = DatabaseManager()
        self.ai_service = DobbyAIService(fireworks_api_key)
        self.application = None
        
        # Состояния пользователей
        self.user_current_group = {}
        self.user_creating_group = {}  # Для хранения названия группы при создании
        self.user_state = {}  # Для отслеживания состояния (ожидание ввода названия группы)
        
        # Доступные языки
        self.LANGUAGES = {
            "en": "🇬🇧 English",
            "ru": "🇷🇺 Русский",
            "es": "🇪🇸 Español",
            "fr": "🇫🇷 Français",
            "de": "🇩🇪 Deutsch",
            "tr": "🇹🇷 Türkçe"
        }

    async def initialize(self):
        """Инициализация бота"""
        await self.db.init_database()
        
        self.application = Application.builder().token(self.bot_token).build()
        
        # Обработчики
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("create_group", self.create_group_command))
        self.application.add_handler(CommandHandler("groups", self.list_groups_command))
        self.application.add_handler(CallbackQueryHandler(self.handle_callback))
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /start с проверкой языковых настроек"""
        user = update.effective_user
        
        # Проверить завершена ли настройка языка
        setup_complete, native_lang = await self.db.get_user_language_setup_status(user.id)
        
        if not setup_complete:
            # Показать выбор базового языка
            await self.show_language_selection(update)
            return
        
        # Создать/получить пользователя
        db_user = await self.db.get_or_create_user(
            telegram_id=user.id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name
        )
        
        welcome_text = f"""
  **Hi, {user.first_name}! I'm Dobby!** 

Welcome to **DobbyLearn** - AI-powered language learning!

✨ **Key Features:**
• 🤖 **AI auto-translation** - just type any word!
• 📝 **AI examples** - unique every time
• 🎯 **Anki system** - smart spaced repetition
• 🌍 **Multi-language** - learn any language
• 📚 **Groups** - organize by topics

📖 **How to use:**
1️⃣ Create a group (or use default)
2️⃣ Just type words in chat:
```
serendipity
resilient, ephemeral
```
3️⃣ Open app → AI already translated everything! 🚀

Your base language: **{self.LANGUAGES.get(native_lang, native_lang)}**
"""
        
        keyboard = [
            [InlineKeyboardButton(
                "🎓 Open DobbyLearn", 
                web_app=WebAppInfo(url=self.webapp_url)
            )],
            [InlineKeyboardButton("➕ Create Group", callback_data="create_group"),
             InlineKeyboardButton("📚 My Groups", callback_data="show_groups")],
            [InlineKeyboardButton("🌍 Change Language", callback_data="change_language")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            welcome_text,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )

    async def show_language_selection(self, update: Update, is_first_time: bool = True):
        """Показать выбор базового языка"""
        if is_first_time:
            text = """
👋 **Welcome to DobbyLearn!**

First, select your **base language** (for translations and explanations):
"""
        else:
            text = """
🌍 **Change base language**

Select your **base language** (for translations and explanations):
"""
        
        # Создать кнопки языков
        keyboard = []
        for lang_code, lang_name in self.LANGUAGES.items():
            keyboard.append([
                InlineKeyboardButton(
                    lang_name,
                    callback_data=f"set_native_lang_{lang_code}"
                )
            ])
        
        
        if not is_first_time:
            keyboard.append([InlineKeyboardButton("◀️ Back to Menu", callback_data="back_to_main")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if update.message:
            await update.message.reply_text(text, parse_mode='Markdown', reply_markup=reply_markup)
        elif update.callback_query:
            await update.callback_query.edit_message_text(
                text, parse_mode='Markdown', reply_markup=reply_markup
            )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /help"""
        help_text = """
  **DobbyLearn Commands:**

**Main:**
• `/start` - Start bot
• `/help` - This help
• `/groups` - Show all groups

**Groups:**
• `/create_group [name]` - Create new group
  Example: `/create_group Spanish Cities`

**Adding words:**
Just send words to chat:
```
adventure, journey, quest
```
or:
```
adventure
journey
quest
```

🤖 **AI will automatically:**
✅ Translate to your base language
✅ Generate unique example sentence
✅ Add pronunciation
✅ Create flashcard

🎓 **Open Mini App to start learning!**
"""
        
        await update.message.reply_text(help_text, parse_mode='Markdown')

    async def list_groups_callback(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Показать группы через callback"""
        user_id = query.from_user.id
        
        # Создать временный update объект для совместимости
        class FakeUpdate:
            def __init__(self, callback_query):
                self.callback_query = callback_query
                self.message = None
                self.effective_user = callback_query.from_user
        
        fake_update = FakeUpdate(query)
        await self.list_groups_command(fake_update, context)
    
    async def list_groups_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать группы с тегами языков"""
        user_id = update.effective_user.id
        
        db_user = await self.db.get_or_create_user(user_id)
        groups = await self.db.get_user_groups(db_user.id)
        
        if not groups:
            await update.message.reply_text(
                "📚 No groups yet.\n"
                "Use `/create_group [name]` to create!"
            )
            return
        
        text = "📚 **Your word groups:**\n\n"
        keyboard = []
        
        for group in groups:
            progress = 0
            if group.total_words > 0:
                progress = int((group.learned_words / group.total_words) * 100)
            
            # Получить тег языка
            native_lang = getattr(group, 'native_language', 'ru')
            target_lang = getattr(group, 'target_language', 'en')
            lang_tag = await self.db.get_language_tag(native_lang, target_lang)
            
            text += f"{group.icon} **{group.name}** {lang_tag}\n"
            text += f"   📖 Words: {group.total_words} | ✅ Learned: {group.learned_words} ({progress}%)\n\n"
            
            keyboard.append([
                InlineKeyboardButton(
                    f"{group.icon} {group.name} {lang_tag}",
                    callback_data=f"select_group_{group.id}"
                )
            ])
        
        keyboard.extend([
            [InlineKeyboardButton("➕ Create Group", callback_data="create_group")],
            [InlineKeyboardButton("🎓 Open App", 
                                web_app=WebAppInfo(url=self.webapp_url))],
            [InlineKeyboardButton("◀️ Back to Menu", callback_data="back_to_main")]
        ])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Проверяем есть ли update.message или callback query
        if update.message:
            await update.message.reply_text(text, parse_mode='Markdown', reply_markup=reply_markup)
        elif update.callback_query:
            await update.callback_query.message.reply_text(text, parse_mode='Markdown', reply_markup=reply_markup)

    async def create_group_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Создать группу - сначала выбор языка изучения"""
        user_id = update.effective_user.id
        
        if not context.args:
            await update.message.reply_text(
                "📚 **Create a group**\n\n"
                "Usage: `/create_group [name]`\n\n"
                "**Examples:**\n"
                "• `/create_group Spanish Cities`\n"
                "• `/create_group Movie Vocabulary`"
            )
            return
        
        group_name = " ".join(context.args)
        
        # Сохранить название группы в состоянии
        self.user_creating_group[user_id] = group_name
        
        # Показать выбор языка изучения
        await self.show_target_language_selection(update, group_name)

    async def show_target_language_selection(self, update, group_name: str):
        """Показать выбор языка изучения для группы"""
        user_id = update.effective_user.id if update.effective_user else None
        
        # Получить базовый язык пользователя
        if user_id:
            _, native_lang = await self.db.get_user_language_setup_status(user_id)
        else:
            native_lang = "ru"
        
        text = f"""
📚 Creating group: **{group_name}**

Select the **language you're learning** in this group:
"""
        
        keyboard = []
        for lang_code, lang_name in self.LANGUAGES.items():
            if lang_code != native_lang:  # Не показывать базовый язык
                keyboard.append([
                    InlineKeyboardButton(
                        lang_name,
                        callback_data=f"set_target_lang_{lang_code}"
                    )
                ])
        
        # Добавить кнопку отмены
        keyboard.append([InlineKeyboardButton("❌ Cancel", callback_data="back_to_main")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if update.message:
            await update.message.reply_text(text, parse_mode='Markdown', reply_markup=reply_markup)
        elif update.callback_query:
            await update.callback_query.edit_message_text(
                text, parse_mode='Markdown', reply_markup=reply_markup
            )

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка кнопок"""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        
        # Установка базового языка
        if query.data.startswith("set_native_lang_"):
            lang_code = query.data.replace("set_native_lang_", "")
            
            db_user = await self.db.get_or_create_user(user_id)
            await self.db.update_user_language_setup(db_user.id, lang_code)
            
            # Обновить язык во всех существующих группах пользователя
            await self.db.update_all_user_groups_language(db_user.id, lang_code)
            
            keyboard = [
                [InlineKeyboardButton("➕ Create Group", callback_data="create_group")],
                [InlineKeyboardButton("🎓 Open DobbyLearn", 
                                    web_app=WebAppInfo(url=self.webapp_url))],
                [InlineKeyboardButton("◀️ Back to Menu", callback_data="back_to_main")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                f"✅ Base language set: **{self.LANGUAGES[lang_code]}**\n\n"
                f"All your groups updated to translate to {self.LANGUAGES[lang_code]}!\n\n"
                f"Now you can create groups and add words!",
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
        
        # Установка языка изучения для группы
        elif query.data.startswith("set_target_lang_"):
            target_lang = query.data.replace("set_target_lang_", "")
            group_name = self.user_creating_group.get(user_id, "My Words")
            
            db_user = await self.db.get_or_create_user(user_id)
            _, native_lang = await self.db.get_user_language_setup_status(user_id)
            
            # Создать группу
            group = await self.db.create_word_group(
                user_id=db_user.id,
                name=group_name,
                native_language=native_lang,
                target_language=target_lang
            )
            
            self.user_current_group[user_id] = group.id
            
            lang_tag = await self.db.get_language_tag(native_lang, target_lang)
            
            keyboard = [
                [InlineKeyboardButton("🎓 Open DobbyLearn", 
                                    web_app=WebAppInfo(url=self.webapp_url))],
                [InlineKeyboardButton("➕ Create Another", callback_data="create_group"),
                 InlineKeyboardButton("📚 My Groups", callback_data="show_groups")],
                [InlineKeyboardButton("◀️ Back to Menu", callback_data="back_to_main")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                f"✅ Group **{group_name}** {lang_tag} created!\n\n"
                f"Learning: **{self.LANGUAGES[target_lang]}**\n"
                f"Translations: **{self.LANGUAGES[native_lang]}**\n\n"
                f"Now send words and AI will handle the rest! 🤖",
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
            
            # Очистить состояние
            if user_id in self.user_creating_group:
                del self.user_creating_group[user_id]
        
        elif query.data == "show_groups":
            await self.list_groups_callback(query, context)
        
        elif query.data == "change_language":
            await self.show_language_selection(update, is_first_time=False)
        
        elif query.data.startswith("select_group_"):
            group_id = int(query.data.split("_")[2])
            self.user_current_group[user_id] = group_id
            
            keyboard = [
                [InlineKeyboardButton("🎓 Open DobbyLearn", 
                                    web_app=WebAppInfo(url=self.webapp_url))],
                [InlineKeyboardButton("◀️ Back to Menu", callback_data="back_to_main")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                f"✅ Group selected!\n\nAll new words will be added to this group.",
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
        
        elif query.data == "back_to_main":
            # Вернуться в главное меню
            # Очистить состояния
            if user_id in self.user_state:
                del self.user_state[user_id]
            if user_id in self.user_creating_group:
                del self.user_creating_group[user_id]
            
            # Показать главное меню
            user = query.from_user
            db_user = await self.db.get_or_create_user(user_id)
            _, native_lang = await self.db.get_user_language_setup_status(user_id)
            
            welcome_text = f"""
  **Hi, {user.first_name}! I'm Dobby!** 

Welcome to **DobbyLearn** - AI-powered language learning!

✨ **Key Features:**
• 🤖 **AI auto-translation** - just type any word!
• 📝 **AI examples** - unique every time
• 🎯 **Anki system** - smart spaced repetition
• 🌍 **Multi-language** - learn any language
• 📚 **Groups** - organize by topics

📖 **How to use:**
1️⃣ Create a group (or use default)
2️⃣ Just type words in chat:
```
serendipity
resilient, ephemeral
```
3️⃣ Open app → AI already translated everything! 🚀

Your base language: **{self.LANGUAGES.get(native_lang, native_lang)}**
"""
            
            keyboard = [
                [InlineKeyboardButton(
                    "🎓 Open DobbyLearn", 
                    web_app=WebAppInfo(url=self.webapp_url)
                )],
                [InlineKeyboardButton("➕ Create Group", callback_data="create_group"),
                 InlineKeyboardButton("📚 My Groups", callback_data="show_groups")],
                [InlineKeyboardButton("🌍 Change Language", callback_data="change_language")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                welcome_text,
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
        
        elif query.data == "create_group":
            # Установить состояние ожидания названия группы
            self.user_state[user_id] = "awaiting_group_name"
            
            keyboard = [
                [InlineKeyboardButton("❌ Cancel", callback_data="back_to_main")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                "📚 **Create a new group**\n\n"
                "📝 Please type the group name:\n\n"
                "**Examples:**\n"
                "• Spanish Movies\n"
                "• Daily Words\n"
                "• Tech Vocabulary",
                parse_mode='Markdown',
                reply_markup=reply_markup
            )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка текста - добавление слов с AI переводом"""
        user_id = update.effective_user.id
        text = update.message.text.strip()
        
        if not text:
            return
        
        # Получить пользователя
        db_user = await self.db.get_or_create_user(user_id)
        
        # Проверить настройку языка
        setup_complete, native_lang = await self.db.get_user_language_setup_status(user_id)
        if not setup_complete:
            await self.show_language_selection(update)
            return
        
        # Проверить состояние - ожидание названия группы
        if self.user_state.get(user_id) == "awaiting_group_name":
            # Сохранить название и показать выбор языка изучения
            self.user_creating_group[user_id] = text
            del self.user_state[user_id]
            
            await self.show_target_language_selection(update, text)
            return
        
        # Получить текущую группу
        group_id = self.user_current_group.get(user_id)
        if not group_id:
            groups = await self.db.get_user_groups(db_user.id)
            if groups:
                group_id = groups[0].id
                self.user_current_group[user_id] = group_id
        
        # Получить язык группы (И native_lang из группы!)
        groups = await self.db.get_user_groups(db_user.id)
        target_lang = "en"  # По умолчанию английский
        native_lang = "ru"  # По умолчанию русский
        for g in groups:
            if g.id == group_id:
                target_lang = g.target_language
                native_lang = g.native_language  # БРАТЬ native_lang ИЗ ГРУППЫ!
                break
        
        # Распарсить слова
        words = await self.parse_words(text)
        
        if not words:
            await update.message.reply_text("❌ Could not parse words. Try again!")
            return
        
        # Показать статус
        status_msg = await update.message.reply_text(
            f"🤖 AI is processing {len(words)} word(s)...\n"
            f"📖 Words in {self.LANGUAGES[target_lang]}\n"
            f"📝 Translating to {self.LANGUAGES[native_lang]}..."
        )
        
        # Добавить слова с AI переводом
        added_count = 0
        for word in words:
            try:
                # Получить AI перевод + пример
                # Слово на target_lang (язык группы), перевод на native_lang
                translation_data = await self.ai_service.translate_word(
                    word=word,
                    source_language=target_lang,  # Язык СЛОВА (испанский, английский)
                    target_language=native_lang   # Язык ПЕРЕВОДА (русский)
                )
                
                await self.db.add_word(
                    user_id=db_user.id,
                    word_data={
                        'word': word,
                        'translation': translation_data.get('translation'),
                        'added_via': 'chat'
                    },
                    group_id=group_id
                )
                added_count += 1
            except Exception as e:
                logger.error(f"Ошибка добавления слова {word}: {e}")
        
        if added_count > 0:
            await status_msg.edit_text(
                f"✅ Added **{added_count}** word(s)!\n\n"
                f"🤖 AI translated to {self.LANGUAGES[native_lang]}\n"
                f"📝 Examples generated\n\n"
                f"🎓 Open app to start learning!",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("🎓 Open DobbyLearn", 
                                       web_app=WebAppInfo(url=self.webapp_url))
                ]])
            )
        else:
            await status_msg.edit_text("❌ Could not add words. Try again!")

    async def parse_words(self, text: str) -> List[str]:
        """Распарсить слова из текста"""
        words = []
        
        if ',' in text:
            parts = text.split(',')
            for part in parts:
                word = part.strip().lower()
                if word and len(word) > 1:
                    words.append(word)
        elif '\n' in text:
            parts = text.split('\n')
            for part in parts:
                word = part.strip().lower()
                if word and len(word) > 1:
                    words.append(word)
        else:
            word = text.strip().lower()
            if word and len(word) > 1:
                words.append(word)
        
        return words

def main():
    import os
    load_dotenv()
    
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    fireworks_api_key = os.getenv("FIREWORKS_API_KEY", "")
    webapp_url = os.getenv("WEBAPP_URL", "https://yourdomain.com")
    
    if not bot_token:
        raise ValueError("TELEGRAM_BOT_TOKEN не найден в .env")
    
    logger.info("DobbyLearn Bot with AI starting...")
    
    # Создать event loop и запустить инициализацию
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    bot = DobbyLearnBot(bot_token, fireworks_api_key, webapp_url)
    loop.run_until_complete(bot.initialize())
    
    logger.info("✅ Bot initialized, starting polling...")
    
    # Запустить polling 
    bot.application.run_polling()

if __name__ == "__main__":
    main()
