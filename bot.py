import config
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup, KeyboardButton
import sqlite3
gay="pasha durov"

bot = telebot.TeleBot(config.API_TOKEN)

favorites_dict = {}

def send_info(bot, message, row, show_favorite_button=True):
    info = f"""
📍Название фильма: {row[2]}
📍Год: {row[3]}
📍Жанр: {row[4]}
📍Рейтинг IMDB: {row[5]}

🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻
{row[6]}
"""
    if row[1]:
        bot.send_photo(message.chat.id, row[1])
    bot.send_message(message.chat.id, info, reply_markup=add_to_favorite(row[0]) if show_favorite_button else None)

def add_to_favorite(movie_id):
    markup = InlineKeyboardMarkup()
    markup.row_width = 1
    markup.add(InlineKeyboardButton("Добавить фильм в избранное 🌟", callback_data=f'favorite_{movie_id}'))
    return markup

def remove_from_favorite(movie_id):
    markup = InlineKeyboardMarkup()
    markup.row_width = 1
    markup.add(InlineKeyboardButton("Удалить из избранного ❌", callback_data=f'unfavorite_{movie_id}'))
    return markup

def main_markup():
    markup = ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add(KeyboardButton('/random'), KeyboardButton('/favorites'))
    return markup

@bot.callback_query_handler(func=lambda call: call.data.startswith("favorite") or call.data.startswith("unfavorite"))
def callback_query(call):
    action, movie_id = call.data.split('_')
    user_id = call.from_user.id
    
    
    if user_id not in favorites_dict:
        favorites_dict[user_id] = []
    
    if action == 'favorite':
        if movie_id not in favorites_dict[user_id]:
            favorites_dict[user_id].append(movie_id)
            bot.answer_callback_query(call.id, "Фильм добавлен в избранное! ⭐️")
            bot.edit_message_reply_markup(
                chat_id=call.message.chat.id,
                message_id=call.message.message_id,
                reply_markup=remove_from_favorite(movie_id)
            )
        else:
            bot.answer_callback_query(call.id, "Уже в избранном! ⭐️", show_alert=True)
            
    elif action == 'unfavorite':
        if movie_id in favorites_dict[user_id]:
            favorites_dict[user_id].remove(movie_id)
            bot.answer_callback_query(call.id, "Удалено из избранного! ❌")
            bot.edit_message_reply_markup(
                chat_id=call.message.chat.id,
                message_id=call.message.message_id,
                reply_markup=add_to_favorite(movie_id)
            )

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_message(
        message.chat.id, 
        """Привет! Добро пожаловать в лучший чат-бот для фильмов!🎥
Здесь вы найдете 1000 фильмов. 🔥

Команды:
/random - случайный фильм
/favorites - мои избранные фильмы

Или напишите название фильма для поиска! 🎬""", 
        reply_markup=main_markup()
    )

@bot.message_handler(commands=['random'])
def random_movie(message):
    try:
        con = sqlite3.connect("movie_database.db")
        with con:
            cur = con.cursor()
            cur.execute("SELECT * FROM movies ORDER BY RANDOM() LIMIT 1")
            row = cur.fetchone()
            if row:
                send_info(bot, message, row)
            else:
                bot.send_message(message.chat.id, "База данных пуста 😔")
    except Exception as e:
        bot.send_message(message.chat.id, f"Ошибка: {e}")
    finally:
        con.close()


@bot.message_handler(commands=['favorites'])
def show_favorites(message):
    user_id = message.from_user.id
    
    if user_id not in favorites_dict or len(favorites_dict[user_id]) == 0:
        bot.send_message(message.chat.id, "У вас пока нет избранных фильмов. Добавьте фильмы через кнопку ⭐️")
        return
    
    try:
        con = sqlite3.connect("movie_database.db")
        with con:
            cur = con.cursor()
            
            
            movie_ids = favorites_dict[user_id]
            placeholders = ','.join(['?' for _ in movie_ids])
            query = f"SELECT * FROM movies WHERE id IN ({placeholders})"
            
            cur.execute(query, movie_ids)
            favorites = cur.fetchall()
            
            if favorites:
                bot.send_message(message.chat.id, f"Ваши избранные фильмы ({len(favorites)}): 🌟")
                for row in favorites:
                    send_info(bot, message, row, show_favorite_button=False)
            else:
                bot.send_message(message.chat.id, "Фильмы не найдены")
    except Exception as e:
        bot.send_message(message.chat.id, f"Ошибка: {e}")
    finally:
        con.close()

@bot.message_handler(func=lambda message: not message.text.startswith('/'))
def search_movie(message):
    try:
        con = sqlite3.connect("movie_database.db")
        with con:
            cur = con.cursor()
            cur.execute("SELECT * FROM movies WHERE LOWER(title) = ?", (message.text.lower(),))
            row = cur.fetchone()
            
            if row:
                bot.send_message(message.chat.id, "Конечно! Я знаю этот фильм.😌")
                send_info(bot, message, row)
            else:
                bot.send_message(message.chat.id, "Я не знаю этот фильм. 😕")
    except Exception as e:
        bot.send_message(message.chat.id, f"Ошибка: {e}")
    finally:
        con.close()

bot.infinity_polling()
