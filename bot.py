import argparse, json, os, traceback, subprocess, sys
from openai import OpenAI
from dotenv import dotenv_values
from datetime import datetime
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, ConversationHandler, CallbackQueryHandler, MessageHandler, filters
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup

sys.path.insert(0, "core")
from core.utils import YEAR_INFO, COUNTRY_INFO

LOG_FILE = 'log.txt'

MESSAGES = []

CLIENT = OpenAI(
    api_key=dotenv_values()['OPENAI_API_KEY']
)

HELP_MSG = """
    You can control me by sending these commands:
    /start - Start the bot
    /run - Run the scraper
    /status - Check the status of the scraper
    /info - Get information about countries and years
    /logs - Show the log file of the latest scraper run
    /clear - Clear chatbot messages
    /cancel - Cancel the operation
    /help - Get help
"""

def recursive_listdir(root_dir:str, file_extension:str='.csv') -> dict:
    files = {}
    for root, _, filenames in os.walk(root_dir):
        data_type = root.split('/')[-1]
        files[data_type] = {}
        for filename in filenames:
            if filename.endswith(file_extension):
                reduced_name = filename.replace(file_extension, '').replace(data_type, '').strip('_')
                year = reduced_name.split('_')[-1]
                if year not in files[data_type]:
                    files[data_type][year] = []
                reduced_name = reduced_name.replace(f"_{year}", '')
                files[data_type][year].append(reduced_name)
    if '' in files:
        del files['']
    return files

def check_run_status() -> tuple:
    running = False
    command_line = ''
    try:
        ps = subprocess.Popen("ps -ef | grep scraper.py", shell=True, stdout=subprocess.PIPE)
        output = ps.stdout.read()
        ps.stdout.close()
        ps.wait()
        if len(output) > 0:
            for command in output.decode().split('\n'):
                if '--headless' in command:
                    running = True
                    command_line = command
                    break
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        return running, command_line

def run_scraper_in_background(config_path, data_type, data_dir, countries, years) -> bool:
    success = False
    try:
        python_command = "python scraper.py {} --headless --download {} --data_dir {} --countries {} --years {}".format(config_path, data_type, data_dir, countries, years)
        os.system(f"nohup {python_command} > {LOG_FILE} 2>&1 &")
        success = True
    except Exception as e:
        print(f"Error: {e}")
    finally:
        return success

# Define states for conversation
ASKING_TYPE, ASKING_YEAR, ASKING_COUNTRIES, ASKING_EXECUTION = range(4)

# Command handler for the /start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Define the custom keyboard layout
    keyboard = [
        [KeyboardButton('/start'), KeyboardButton('/run'), KeyboardButton('/status'), KeyboardButton('/logs')], 
        [KeyboardButton('/info'), KeyboardButton('/clear'), KeyboardButton('/cancel'), KeyboardButton('/help')]
    ]
    
    # Create a ReplyKeyboardMarkup object with the keyboard layout
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

    # Send a welcome message with the custom keyboard
    await update.message.reply_text("""
        Welcome to the ENTSOE bot! 🤖
        The purpose of this bot is to ease data collection from the ⚡ ENTSOE ⚡ transparency platform.
        Use the buttons ⌨️ below to navigate or ask for /help. You can also chat with me! 🗨️
        """, reply_markup=reply_markup
    )

# Command handler for the /info command
async def show_info(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    inline_keyboard = [
        [InlineKeyboardButton("Years", callback_data='years')],
        [InlineKeyboardButton("Countries", callback_data='countries')]
    ]
    reply_markup = InlineKeyboardMarkup(inline_keyboard)
    await update.message.reply_text('Which parameter do you want to know about?', reply_markup=reply_markup)

async def show_info_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    if query.data == 'years':
        await query.edit_message_text(YEAR_INFO)
    elif query.data == 'countries':
        await query.edit_message_text(COUNTRY_INFO)

# Command handler for the /logs command
async def show_logs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            logs = f.read()
        k = 10
        last_rows = logs.split('\n')[-k:]
        await update.message.reply_text("""
        Last {} lines of the log file:
        {}""".format(k, '\n'.join(last_rows)))
    else:
        await update.message.reply_text("Log file not found. ❌")

# Command handler for the /status command
async def show_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    running, command_line = check_run_status()
    if running:
        await update.message.reply_text(f"Scraper is running with the following command line:\n {command_line}")
    else:
        await update.message.reply_text("Scraper is not running. You can start new operations. ✅")

# Command handler for the /run command
async def ask_data_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # Inline keyboard
    inline_keyboard = [
        [InlineKeyboardButton("Total load", callback_data='total_load')],
        [InlineKeyboardButton("Power generation", callback_data='generator')],
        [InlineKeyboardButton("Border transmission", callback_data='border')]
    ]
    reply_markup = InlineKeyboardMarkup(inline_keyboard)
    await update.message.reply_text('Select the data type that you want to crawl:', reply_markup=reply_markup)
    return ASKING_TYPE

async def ask_data_type_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    context.user_data['data_type'] = query.data
    await query.answer()
    await query.edit_message_text(f"Selected data type: {context.user_data['data_type']}")
    await ask_year(update, context)  # Directly call the ask_year function
    return ASKING_YEAR

async def ask_year(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.message.reply_text('Enter the year that you want to crawl:')
    return ASKING_YEAR

async def ask_year_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    year = update.message.text
    context.user_data['year'] = year
    await ask_countries(update, context)
    return ASKING_COUNTRIES

async def ask_countries(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Enter the countries that you want to crawl:')
    return ASKING_COUNTRIES

async def ask_countries_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    countries = update.message.text
    context.user_data['countries'] = countries

    # summarize the data
    data_type = context.user_data['data_type']
    year = context.user_data['year']
    countries = context.user_data['countries']
    data_dir = context.bot_data['data_dir']
    await update.message.reply_text(f"Data type: {data_type}\nYear: {year}\nCountries: {countries}\nData directory: {data_dir}")
    await ask_execution(update, context)
    return ASKING_EXECUTION

async def ask_execution(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    inline_keyboard = [
        [InlineKeyboardButton("Yes", callback_data='yes')],
        [InlineKeyboardButton("No", callback_data='no')]
    ]
    reply_markup = InlineKeyboardMarkup(inline_keyboard)
    await update.message.reply_text('Do you want to start the operation?', reply_markup=reply_markup)
    return ASKING_EXECUTION

async def ask_execution_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    if query.data == 'yes':
        running, _ = check_run_status()
        if running:
            msg = "Scraper is already running. Please wait until it finishes. ⌛"
        else:
            # Run the scraper
            data_type = context.user_data['data_type']
            year = context.user_data['year']
            countries = context.user_data['countries']
            data_dir = context.bot_data['data_dir']
            config_path = context.bot_data['config_path']
            start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            success = run_scraper_in_background(config_path, data_type, data_dir, countries, year)
            if success:
                msg = """
                Scraper was successfully started at {} with the following parameters: ✅
                Data type: {}
                Year: {}
                Countries: {}
                """.format(start_time_str, data_type, year, countries)
            else:
                msg = "Failed to start the scraper. ❌"
        await query.message.edit_text(msg)
    else:
        await query.message.edit_text("Operation cancelled. ✅")
    return ConversationHandler.END

# Command handler for the /help command
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_MSG)

# Handle cancellation
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Operation cancelled. ✅")
    return ConversationHandler.END

async def chatgpt_response(messages:list, model_name: str, max_tokens: int, temperature: float, verbose:bool=False) -> str:
    if verbose:
        print(f"Calling OpenAI with model {model_name}, max tokens: {max_tokens}, temperature: {temperature} and messages:")
        for msg in messages:
            print(f"{msg['role']}: {msg['content']}")
    try:
        chat_completion = CLIENT.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        #print(chat_completion)
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "Sorry, I couldn't process your request at the moment."

async def handle_random_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text
    scraped_files = recursive_listdir(context.bot_data['data_dir'])
    #print(f"User message: {user_message}")
    new_msg = """
    Please respond to the user message based on the information provided.
    
    # User message: %s
    
    # Help information:
    %s

    # List of countries available for scraping with their ENTSOE abbreviations:
    %s

    # Dictionary of scraped countries by year or data type:
    %s
    """ % (user_message, HELP_MSG, COUNTRY_INFO, scraped_files)
    if len(MESSAGES) == 0:
        MESSAGES.append({"role": "system", "content": "You are a helpful Telegram bot assistant. You can understand casual language including emojis, abbreviations and typos. Your answer should be as short as possible to easily fit on small screens."})
    MESSAGES.append({"role": "user", "content": new_msg})
    openai_config = context.bot_data.get('openai_config', {})
    response = await chatgpt_response(
        MESSAGES,
        openai_config.get('model_name', 'gpt-3.5-turbo'),
        openai_config.get('max_tokens', 100),
        openai_config.get('temperature', 0.5),
        verbose=True
    )
    MESSAGES.append({"role": "assistant", "content": response})
    await update.message.reply_text(response)

async def clear_messages(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    MESSAGES.clear()
    await update.message.reply_text("Chatbot messages were cleared. ✅")

def main(config_path, data_dir):
    # Load the configuration file
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Store the data directory in bot_data
    bot_data = {
        'data_dir': data_dir,
        'config_path': config_path,
        'openai_config': {
            'model_name': config["openai"]["model_name"],
            'max_tokens': config["openai"]["max_tokens"],
            'temperature': config["openai"]["temperature"]
        }
    }
    
    if "telegram" in config:
        tg_config = config["telegram"]
        if "token" in tg_config:
            # Create the application
            app = ApplicationBuilder().token(tg_config["token"]).build()

            # Set bot_data
            app.bot_data.update(bot_data)

            # Define the conversation handler
            conv_handler = ConversationHandler(
                entry_points=[CommandHandler('run', ask_data_type)],
                states={
                    ASKING_TYPE: [CallbackQueryHandler(ask_data_type_callback)],
                    ASKING_YEAR: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_year_callback)],
                    ASKING_COUNTRIES: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_countries_callback)],
                    ASKING_EXECUTION: [CallbackQueryHandler(ask_execution_callback)]
                },
                fallbacks=[CommandHandler('cancel', cancel)]
            )
            
            # Add the conversation handler to the application
            app.add_handler(conv_handler)
            
            # Add command handlers to the application
            app.add_handler(CommandHandler('start', start))
            app.add_handler(CommandHandler('info', show_info))
            app.add_handler(CallbackQueryHandler(show_info_callback))
            app.add_handler(CommandHandler('status', show_status))
            app.add_handler(CommandHandler('logs', show_logs))
            app.add_handler(CommandHandler('clear', clear_messages))
            app.add_handler(CommandHandler('help', help_command))

            # Add a handler for random text messages
            app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_random_message))
  
            # Run the bot
            app.run_polling()
        else:
            raise ValueError("Telegram token not found in the configuration file")
    else:
        raise ValueError("Telegram configuration not found in the configuration file")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Telegram bot for Entsoe data')
    parser.add_argument('config_path', type=str, help='Path to the JSON configuration file')
    parser.add_argument('data_dir', type=str, help='Directory to save downloaded files')
    args = parser.parse_args()
    #print(recursive_listdir(args.data_dir))
    main(args.config_path, args.data_dir)