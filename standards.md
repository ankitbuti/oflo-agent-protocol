# Python PEP 8 Naming Standards Guide

## General Naming Conventions

### ✅ Preferred
- Use **snake_case** for function and variable names.
- Use **PascalCase** for class names.
- Use **UPPER_SNAKE_CASE** for constants.
- Use **snake_case** for module and package names.

### 🚫 Avoid
- Hyphens (`word-about`) – Use underscores instead.
- Mixed case in variables (`wordAbout`) – Use lowercase with underscores.
- Single-character variable names (unless for loop counters like `i`, `j`).

---

## Naming Rules by Category

### **1. Variables & Functions**
✅ `word_about`
```python
user_count = 10
def calculate_total_price():
    pass
```

🚫 `WordAbout`, `wordAbout`
```python
UserCount = 10  # ❌ Avoid PascalCase for variables
```

---

### **2. Classes**
✅ `WordAbout`
```python
class DataProcessor:
    pass
```

🚫 `word_about`
```python
class data_processor:  # ❌ Avoid snake_case for class names
    pass
```

---

### **3. Constants**
✅ `WORD_ABOUT`
```python
MAX_RETRIES = 5
DATABASE_URL = "https://example.com"
```

🚫 `WordAbout`, `word_about`
```python
maxRetries = 5  # ❌ Avoid camelCase or mixed case for constants
```

---

### **4. Modules & Package Names**
✅ `word_about.py`
```shell
my_module.py
config_loader.py
```

🚫 `WordAbout.py`, `word-about.py`
```shell
WordAbout.py  # ❌ Avoid PascalCase
word-about.py  # ❌ Avoid hyphens
```

---

### **5. File Names**
✅ `word_about.py`
```shell
config_manager.py
user_auth.py
```

🚫 `WordAbout.py`, `word-about.py`
```shell
WordAbout.py  # ❌ Avoid PascalCase
word-about.py  # ❌ Avoid hyphens
```

---

## Summary of Best Practices
| Item            | Convention         | Example        |
|----------------|-------------------|--------------|
| Variables      | `snake_case`       | `user_count` |
| Functions      | `snake_case`       | `calculate_total()` |
| Classes       | `PascalCase`       | `DataProcessor` |
| Constants     | `UPPER_SNAKE_CASE` | `MAX_RETRIES` |
| Modules       | `snake_case.py`    | `config_loader.py` |
| Packages      | `snake_case`       | `data_tools/` |

Following PEP 8 ensures readable and maintainable code that is consistent across Python projects!

# Python PEP 8 Naming Standards Guide

## General Naming Conventions

### ✅ Preferred
- Use **snake_case** for function and variable names.
- Use **PascalCase** for class names.
- Use **UPPER_SNAKE_CASE** for constants.
- Use **snake_case** for module and package names.

### 🚫 Avoid
- Hyphens (`word-about`) – Use underscores instead.
- Mixed case in variables (`wordAbout`) – Use lowercase with underscores.
- Single-character variable names (unless for loop counters like `i`, `j`).

---

## Naming Rules by Category

### **1. Variables & Functions**
✅ `word_about`
```python
user_count = 10
def calculate_total_price():
    pass
```

🚫 `WordAbout`, `wordAbout`
```python
UserCount = 10  # ❌ Avoid PascalCase for variables
```

---

### **2. Classes**
✅ `WordAbout`
```python
class DataProcessor:
    pass
```

🚫 `word_about`
```python
class data_processor:  # ❌ Avoid snake_case for class names
    pass
```

---

### **3. Constants**
✅ `WORD_ABOUT`
```python
MAX_RETRIES = 5
DATABASE_URL = "https://example.com"
```

🚫 `WordAbout`, `word_about`
```python
maxRetries = 5  # ❌ Avoid camelCase or mixed case for constants
```

---

### **4. Modules & Package Names**
✅ `word_about.py`
```shell
my_module.py
config_loader.py
```

🚫 `WordAbout.py`, `word-about.py`
```shell
WordAbout.py  # ❌ Avoid PascalCase
word-about.py  # ❌ Avoid hyphens
```

---

### **5. File Names**
✅ `word_about.py`
```shell
config_manager.py
user_auth.py
```

🚫 `WordAbout.py`, `word-about.py`
```shell
WordAbout.py  # ❌ Avoid PascalCase
word-about.py  # ❌ Avoid hyphens
```

---

## Summary of Best Practices
| Item            | Convention         | Example        |
|----------------|-------------------|--------------|
| Variables      | `snake_case`       | `user_count` |
| Functions      | `snake_case`       | `calculate_total()` |
| Classes       | `PascalCase`       | `DataProcessor` |
| Constants     | `UPPER_SNAKE_CASE` | `MAX_RETRIES` |
| Modules       | `snake_case.py`    | `config_loader.py` |
| Packages      | `snake_case`       | `data_tools/` |

Following PEP 8 ensures readable and maintainable code that is consistent across Python projects!

