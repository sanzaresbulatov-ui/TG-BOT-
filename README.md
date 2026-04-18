# TG-BOT-
--Этот бот способен по картинкам понять на фото человек с грустной или радостной эмоцией бот обучен распозновать эмоции

--1--
import telebot
from telebot import types
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import random , os
from l import get_class
TOKEN = "8333984025:AAFe0QbFReUfJUuvX2cDY7g7vcgqK1Oc7Gs"
bot = telebot.TeleBot(TOKEN)


@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет!я могу определять по фото эмоции просто пришли мне фото и не забдуь в сообщение написать /photo")

@bot.message_handler(content_types=['photo'])
def get_image(message):
    if not message.photo:
        print("No photo")
    file_info = bot.get_file(message.photo[-1].file_id)
    file_name = file_info.file_path.split('/')[-1]
    in_file = bot.download_file(file_info.file_path)
    with open(file_name, "wb") as new_file:  
        new_file.write(in_file)
    res = get_class('keras_model.h5', "labels.txt", file_name)

  
    bot.send_message(message.chat.id, res)
 

bot.polling()


--2--
import telebot
from telebot import types
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import random , os


def get_class(model_path, labels_path, image_path):
    np.set_printoptions(suppress=True)
    model = load_model(model_path, compile=False)
    class_names = open(labels_path, "r", encoding="utf-8"). readlines()
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open (image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names [index]
    confidence_score = prediction[0][index]
    return class_name[2:], confidence_score




