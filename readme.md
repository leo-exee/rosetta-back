# Rosetta API - French Exercise Generator

A FastAPI-based API for generating French as a Foreign Language (FLE) exercises for English-speaking learners, powered by Google's Gemini AI.

## 🚀 Features

- **Intelligent Exercise Generation**: Automatic creation of French exercises adapted to level and context
- **Varied Exercise Types**:
  - 📝 Fill in the blanks
  - 🔗 Definition matcher
  - 📖 Reading comprehension
- **Adaptive Levels**: Beginner, Intermediate, Advanced
- **Thematic Contexts**: Travel, Culture, Kitchen, Literature, Sports, Technology, General
- **Detailed Corrections**: English explanations to facilitate learning
- **REST API**: Modern and documented interface with FastAPI

## 🛠️ Technologies

- **FastAPI**: Modern, high-performance web framework
- **Google Gemini AI**: Artificial intelligence for content generation
- **Pydantic**: Data validation and serialization
- **Python 3.11+**: Programming language
- **Uvicorn**: High-performance ASGI server

## 📋 Prerequisites

- Python 3.11 or higher
- Google Gemini API key
- pip (Python package manager)

## 🔧 Installation

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd rosetta-back
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Environment configuration**

```bash
cp .env.example .env.local  # Create configuration file
```

5. **Configure environment variables**
   Edit `.env.local` and add:

```env
GEMINI_API_KEY=your_gemini_api_key
```

## 🚀 Running the Application

### Development mode

```bash
uvicorn main:app --reload --port 8000
```

### Production mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be accessible at: [http://localhost:8000](http://localhost:8000)

## 📚 API Documentation

Once the application is running, interactive documentation is available:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## 🎯 Usage

### Main Endpoint

```http
POST /exercises
```

### Request Example

```json
{
  "type": "fillInTheBlanks",
  "context": "travel",
  "level": "beginner",
  "count": 3
}
```

### Response Example

```json
{
  "context": "travel",
  "Level": "beginner",
  "exercises": [
    {
      "type": "fillInTheBlanks",
      "text": "Je [...] à la [...] chaque été.",
      "blanks": ["vais", "plage"],
      "answer": "Je vais à la plage chaque été.",
      "blanksCorrection": [
        "'Vais' is the first person singular form of 'aller'",
        "'Plage' refers to the beach in the context of summer vacation"
      ]
    }
  ]
}
```

## 🏗️ Project Structure

```
rosetta-back/
├── app/
│   ├── __init__.py
│   ├── app_v1.py              # FastAPI v1 configuration
│   ├── config/
│   │   └── gemini_config.py   # Gemini AI configuration
│   ├── controllers/
│   │   └── exercise_controller.py  # Exercise controller
│   ├── models/
│   │   ├── error_response.py  # Error models
│   │   ├── exercise_dto.py    # Exercise DTOs
│   │   └── exercise_enum.py   # Enumerations
│   ├── services/
│   │   ├── exercice_service.py    # Business service
│   │   └── gemini_service.py      # Gemini AI service
│   └── utils/
│       ├── json_utils.py      # JSON utilities
│       └── prompt_utils.py    # Prompt construction
├── .vscode/                   # VS Code configuration
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
└── pyproject.toml            # Project configuration
```

## 🔧 Development

### Code Quality Tools

The project uses:

- **Ruff**: Fast linter and formatter
- **Black**: Code formatter
- **MyPy**: Type checking

### Useful Commands

```bash
# Lint with Ruff
ruff check .

# Format with Black
black .

# Type checking with MyPy
mypy .
```

### VS Code Configuration

The project includes an optimized VS Code configuration with:

- Recommended extensions
- Debug configuration
- Auto-formatting on save

## 🌍 Environment Variables

| Variable             | Description           | Required |
| -------------------- | --------------------- | -------- |
| `GEMINI_API_KEY`     | Google Gemini API key | ✅       |
| `SENTRY_ENVIRONMENT` | Sentry environment    | ❌       |

## 🤝 Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## 🆘 Support

For any questions or issues:

- Open an issue on GitHub
- Check the API documentation
- Review application logs

## 🔮 Roadmap

- [ ] Add new exercise types
- [ ] Multi-language support
- [ ] Admin interface
- [ ] Metrics and analytics
- [ ] Automated tests
- [ ] Docker deployment
