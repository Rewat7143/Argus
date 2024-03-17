from typing import Final
from telegram import Update
from telegram.ext import CallbackContext
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

TOKEN: Final = '6983749069:AAEpGRYukJ_VKT1cV036lZr6SFkBhrQiD98'
BOT_USERNAME: Final = '@deepfakedtct_bot'

# Load the model
model = load_model('weights/Meso4_F2F.h5')

# Commands
async def start_command(update: Update, context: CallbackContext):
    await update.message.reply_text('Hola Amigo ! Kaise ho theek ho?')

async def help_command(update: Update, context: CallbackContext):
    await update.message.reply_text('Hey, Kindly upload a photo, so that I can classify it as fake or real.')

async def custom_command(update: Update, context: CallbackContext):
    await update.message.reply_text('This is a custom command, you can add whatever functionality you want')

# Function to classify image using the model
def classify_image(file_path: str):
    img = image.load_img(file_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize image
    prediction = model.predict(img_array)
    return prediction[0][0]  # Assuming it's binary classification

# Handle photo uploads
# Handle photo uploads
async def handle_photo(update: Update, context: CallbackContext):
    message_type: str = update.message.chat.type
    file_id = update.message.photo[-1].file_id
    file_info = await context.bot.get_file(file_id)
    file_path = file_info.file_path

    # Use the bot instance to download the file
    await context.bot.download_file(file_path, f"downloads/{file_id}.jpg")
    print(f'User ({update.message.chat.id}) uploaded photo: "{file_path}"')

    # Classify the image using the model
    prediction = classify_image(f"downloads/{file_id}.jpg")
    response = f"The image is classified as {'Fake' if prediction < 0.5 else 'Real'} (Probability: {prediction:.2f})"
    print('Bot:', response)
    await update.message.reply_text(response)


# Responses
async def handle_message(update: Update, context: CallbackContext):
    message_type: str = update.message.chat.type
    text: str = update.message.text

    print(f'User ({update.message.chat.id}) in {message_type}: "{text}"')

    if message_type == 'group':
        if BOT_USERNAME in text:
            new_text: str = text.replace(BOT_USERNAME, '').strip()
            response: str = handle_response(new_text)
        else:
            response = "Sorry, I can only classify photos right now."
        
        print('Bot:', response)
        await update.message.reply_text(response)

async def error(update: Update, context: CallbackContext):
    print(f'Update {update} caused error {context.error}')

if __name__ == '__main__':
    print('Bot Starting')
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CommandHandler('custom', custom_command))
    
    app.add_handler(MessageHandler(filters._Photo(), handle_photo))
    app.add_handler(MessageHandler(filters.Text(), handle_message))

    app.add_error_handler(error)
    
    print('Bot is running')
    app.run_polling(poll_interval=3)
