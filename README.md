!!! 
ПОДБОР СЕРИАЛА РЕАЛИЗОВАН ТОЛЬКО ПО КАТАЛОГУ ТКК БЕЗ БОЛЬШОЙ БАЗЫ З СЕРИАЛАМИ ИЛИ АПИ Streaming Checker СКОРЕЕ ВСЕГО НЕ БУДЕТ НАХОДИТЬ ПОДОБРАННЫЕ СЕРИАЛЫ ИЗ ТЕХ ЧТО ИИ МНЕ НАКИДАЛ В show_catalog.json (мок данных о доступных сериалах), так что они оба используют этот джейсон
!!!

# DateNight Show Matcher

Рекомендательный агент, который анализирует Instagram-профиль пользователя и подбирает сериалы для совместного просмотра, доступные на Netflix или HBO.

## Архитектура

Граф LangGraph с детерминированным супервайзером:

```
START → supervisor → insta_reader       ─┐
                  → interest_profiler   ─┤→ supervisor → … → END
                  → show_matcher        ─┤
                  → streaming_checker   ─┘
```

| Агент | Модель | Задача |
|---|---|---|
| `insta_reader` | claude-haiku | Извлекает bio, посты и хэштеги из профиля |
| `interest_profiler` | claude-sonnet | Строит психографический профиль пользователя |
| `show_matcher` | claude-sonnet | Выбирает топ-3 сериала из каталога |
| `streaming_checker` | claude-haiku | Фильтрует по активным подпискам (Netflix, HBO) |

Если после фильтрации не осталось ни одного сериала — `show_matcher` перезапускается (до 2 раз).

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install langchain-anthropic langgraph python-dotenv
```

Создайте `.env` в корне проекта:

```
ANTHROPIC_API_KEY=your_key_here
```

## Использование

```bash
python main.py /get-show @art_girl
python main.py /get-show @tech_babe
python main.py @travel_soul
```

### Доступные тестовые профили

| Username | Описание |
|---|---|
| `@art_girl` | Художница, любит мрачное кино и джаз |
| `@tech_babe` | Product designer, фанат sci-fi |
| `@travel_soul` | Цифровой кочевник, любит документалки |

Чтобы добавить новый профиль — отредактируйте `MOCK_INSTAGRAM` в `main.py`.  
Чтобы расширить каталог сериалов — отредактируйте `show_catalog.json`.

## Стриминг

Скрипт выводит ответы в реальном времени:
- прогресс-статусы каждого агента
- токены LLM от `interest_profiler` и `show_matcher` по мере генерации
