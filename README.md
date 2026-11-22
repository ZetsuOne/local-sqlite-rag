# local-sqlite-rag

Локальный гибридный RAG-движок и MCP-сервер на Python и SQLite (`sqlite-vec` + `FTS5`), работающий в связке с локальными серверами инференса (например, LM Studio).

## Возможности
- **Гибридный поиск + RRF:** объединяет векторный поиск по `int8`-эмбеддингам (`sqlite-vec`) и полнотекстовый поиск BM25 (`FTS5`) через алгоритм Reciprocal Rank Fusion.
- **Без тяжёлых фреймворков:** чистый Python, встроенный `sqlite3` и официальные SDK `openai` и `mcp`.
- **Инкрементальная индексация:** отслеживает SHA-256 хэши файлов и фиксирует прогресс после каждого документа — индексацию больших архивов можно безопасно прерывать и продолжать с того же места.
- **Поддержка форматов:** из коробки парсит `.md`, `.txt`, `.pdf`, `.fb2` (включая `.fb2.zip`) и `.epub`.
- **Встроенный MCP-сервер:** предоставляет локальным LLM инструменты `search_knowledge_base` и `list_series` по протоколу `stdio`.

## Установка

```powershell
cd C:\\path\to\local-\sqlite-rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Запустите локальный сервер эмбеддингов с OpenAI-совместимым API (например, LM Studio на `http://localhost:1234/v1` с моделью `text-embedding-bge-m3`).

## Использование

### 1. Индексация документов
```powershell
python ingest.py --docs .\sample_docs --db .\demo.db
```

### 2. Поиск через CLI
```powershell
python search.py --db .\demo.db --query "how does RRF merge search results"
```

Поиск с генерацией ответа через локальную LLM:
```powershell
python search.py --db .\demo.db --query "how does RRF merge search results" --chat-model gemma-4-12b-it
```

## Подключение к LM Studio / Cursor (`mcp.json`)

```json
{
  "mcpServers": {
    "sqlite-rag": {
      "command": "C:\\path\\to\\local-sqlite-rag\\.venv\\Scripts\\python.exe",
      "args": [
        "C:\\path\\to\\local-sqlite-rag\\server.py"
      ],
      "env": {
        "RAG_DB_PATH": "C:\\path\\to\\local-sqlite-rag\\demo.db",
        "LM_STUDIO_URL": "http://localhost:1234/v1",
        "EMBED_MODEL": "text-embedding-bge-m3"
      }
    }
  }
}

```

## Лицензия
Проект распространяется под лицензией [MIT](LICENSE).
