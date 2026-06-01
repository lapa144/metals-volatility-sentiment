# Настройка Telegram парсера

## Шаг 1 — Получи API ключи

1. Зайди на https://my.telegram.org
2. Войди в свой аккаунт Telegram
3. Нажми **API development tools**
4. Заполни форму (название приложения — любое, например "metals_parser")
5. Скопируй **api_id** (число) и **api_hash** (строка)

## Шаг 2 — Вставь ключи в app_code.js

Открой `telegram/app_code.js` и замени:
```js
const apiId = 00000000;      // ← сюда api_id (число)
const apiHash = '';           // ← сюда api_hash (строка)
const pnumber = '';           // ← твой номер телефона (+79...)
const tfapass = '';           // ← пароль двухфакторки (если есть, иначе оставь '')
```

## Шаг 3 — Настрой каналы

Открой `telegram/channels_config.js` и отредактируй список `CHANNELS`.
Добавь или убери каналы под свои нужды.

## Шаг 4 — Установи зависимости и запусти

```bash
cd telegram
npm install
node app_code.js
```

При первом запуске Telegram попросит код подтверждения — введи его в терминал.
Сессия сохраняется в `ses_tele/`, повторный вход не нужен.

## Шаг 5 — Обработай сообщения через NLP

После накопления сообщений в `messages.db`:
```bash
cd src/nlp
pip install -r requirements.txt
python nlp_processor.py
```

## Структура базы данных (messages.db)

| Таблица    | Поля                                                      |
|------------|-----------------------------------------------------------|
| `messages` | id, channel_id, message_id, message, message_object, processed |
| `analysis` | id, message_id, lang, topic, sentiment, assets            |
