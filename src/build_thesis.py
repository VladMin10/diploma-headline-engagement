"""Збірка готового .docx з чернеток розділів за шаблоном кафедри.

Запуск:
    .venv/bin/python -m src.build_thesis

Шаблон `docs/reference/Шаблон магістерської роботи.dotm` використовується
як стартовий документ, а не як довідка: з нього успадковуються всі
іменовані стилі та параметри сторінки. Стилі не створюються — тільки
застосовуються за іменами.

Одна технічна перешкода. `python-docx` відмовляється відкривати `.dotm`,
бо тип вмісту головної частини оголошений як шаблон з макросами.
Тому шаблон спершу перепаковується в тимчасовий `.docx`: у файлі
`[Content_Types].xml` тип головної частини змінюється на документний,
частини з макросами вилучаються, вміст тіла очищується зі збереженням
`sectPr`. Стилі, нумерація, тема і колонтитули лишаються недоторканими.

Зміст, а також нумерація рисунків і таблиць реалізовані **полями Word**,
а не текстом. Після відкриття файл треба оновити (див. інструкцію,
яку скрипт друкує наприкінці).
"""

from __future__ import annotations

import copy
import re
import zipfile
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm

TEMPLATE = Path("docs/reference/Шаблон магістерської роботи.dotm")
SCRATCH = Path("build/_template.docx")
OUT = Path("build/thesis.docx")
FIGURES = Path("reports/figures")

# Стилі, які скрипт застосовує. Перевіряються в шаблоні перед збіркою:
# якщо якогось немає, збірка припиняється, а не підставляє схожий.
REQUIRED_STYLES = [
    "Header_Level_1",
    "Header_Level_2",
    "Header_Level_3",
    "Main_Text",
    "Marked_List",
    "Numbered_List",
    "Figure_Title",
    "Table_Title",
    "References",
]

# Відповідність позначки в чернетці стилю шаблону.
MARKER_STYLE = {
    "Header_Level_1": "Header_Level_1",
    "Header_Level_2": "Header_Level_2",
    "Header_Level_3": "Header_Level_3",
    "Main_Text": "Main_Text",
    "Marked_List": "Marked_List",
    "Numbered_List": "Numbered_List",
    "Table_Title": "Table_Title",
    "Figure_Title": "Figure_Title",
}

# Чернетки в порядку розміщення. `stop` — рядок, з якого починається
# службовий хвіст файлу і який у документ не входить.
PARTS = [
    ("docs/introduction.md", None),
    ("docs/chapter1_analytical.md", "## Список джерел"),
    ("docs/chapter2_research.md", None),
    ("docs/chapter3_results.md", "## Рисунки розділу"),
    ("docs/conclusions.md", None),
]

# Відповідність номера рисунка у тексті файлу зображення.
# Береться з таблиці відповідності в чернетці розділу 3.
FIGURE_FILES = {
    "3.1": "fig10_phase5_configurations.png",
    "3.2": "fig09_phase5_delta_rho.png",
    "3.3": "fig12_exp006_tau.png",
    "3.4": "fig11_exp006_importance.png",
}

PLACEHOLDER = "ЗАПОВНИТИ ВРУЧНУ"

MANUAL_STEPS = """
=== ЩО ЛИШИЛОСЯ ЗРОБИТИ ВРУЧНУ У WORD ===

Скрипт зібрав текст, таблиці, рисунки й поля. Решта потребує Word
або даних, яких у проєкті немає.

1. ТИТУЛЬНИЙ АРКУШ І ЗАВДАННЯ. Бланки скопійовано з шаблону разом
   з таблицями підписів і жовтим підсвіченням. Вони лежать усередині
   елемента керування вмістом на початку шаблону, тому в звичайному
   переліку абзаців не видно; скрипт дістає їх звідти.

   Скрипт підставив чотири значення: тему роботи і два місця з іменем
   здобувача. Решту заповнити вручну за жовтим підсвіченням і зняти
   зафарбовування. Особливо звірити:
   - групу здобувача (у бланку лишилося значення зі зразка);
   - прізвище керівника роботи і завідувача кафедри: у шаблоні в обох
     полях стоїть те саме ім'я, і принаймні одне з них помилкове;
   - рік у рядках дат;
   - рядок консультанта: за рішенням від 8 серпня 2026 року консультант
     іноземної мови не потрібен, тому поле лишається порожнім.

   Дані, яких у проєкті немає і які треба взяти в кафедри: номер і дата
   наказу про затвердження теми, дата видачі завдання, термін подання.

2. АНОТАЦІЯ І ABSTRACT. Заглушки. Анотація - 1 сторінка, склад полів
   у docs/thesis_structure.md розділ 10. Abstract - 10-15 сторінок,
   бо захист англійською; це не переклад анотації, а самостійний переказ
   роботи. Терміни брати з docs/glossary.md. Рядок «Загальний обсяг
   роботи: N сторінок, N рисунків, N посилань» заповнюється останнім,
   коли верстка завершена.

3. ДОДАТКИ. Скрипт створив п'ять заголовків із заглушками. Вміст
   вставити вручну: рубрику з annotations/rubric_v2.md, інструкцію
   анотатора з annotations/prompt_v1.txt, код і таблиці з відповідних
   папок. Код вставляти в таблицю 1x1, як у зразку минулорічної роботи.

4. АПРОБАЦІЯ. У вступі й у підрозділі 3.9 стоїть позначка
   [TODO: публікація]. Заповнити після подання матеріалу і внести
   бібліографічний опис публікації до списку джерел.

5. ПЕРЕХРЕСНІ ПОСИЛАННЯ НА ДЖЕРЕЛА. Скрипт переніс посилання як текст
   у квадратних дужках: [1], [12], [4, 21]. Процедура вимагає робити їх
   інструментом «Перехресні посилання». Це ручна робота: для кожного
   посилання Вставлення -> Перехресне посилання -> тип «Абзац
   з нумерацією» -> потрібна позиція списку. До заміни номери правильні,
   але при додаванні джерела не оновляться самі.

6. НУМЕРАЦІЯ СТОРІНОК. Шаблон містить поле PAGE у нижньому колонтитулі;
   перевірити, що перша сторінка без номера.

7. РОЗРИВИ Й ВИСЯЧІ РЯДКИ. Перевірити, що заголовки не залишилися
   в кінці сторінки окремо від тексту і що таблиці не розірвані
   між сторінками посередині.

8. ОБСЯГ. Перевірити, що основний текст (вступ, три розділи, висновки)
   становить 50-70 сторінок без титульного аркуша, завдання, анотації,
   списку джерел і додатків. У чернетках розділів 2 і 3 вказано,
   що скорочувати першим, якщо обсяг перевищено.

=== ПРО ЗМІСТ І НУМЕРАЦІЮ ПІДПИСІВ ===

Зміст вставлено полем TOC, а не текстом. Номери рисунків і таблиць
вставлені полями SEQ, тому нумерація автоматична: додавання таблиці
всередину розділу перенумерує решту саму.

Рівні структури заголовків перевірено, і тут є важлива обставина.
У шаблоні рівень структури заданий лише для стилю Header_Level_1
(рівень 0, успадкований від «Heading 1»). Стилі Header_Level_2
і Header_Level_3 рівня не мають: вони побудовані на звичайному тексті.
Тому поле з одним перемикачем \\o "1-3" зібрало б лише назви розділів,
а підрозділи в зміст не потрапили б. Скрипт додав перемикач \\t, який
відображає назви цих двох стилів у рівні 2 і 3 змісту:

    TOC \\o "1-3" \\t "Header_Level_2,2,Header_Level_3,3" \\h \\z \\u

Стилі шаблону при цьому не змінювалися. Альтернативою було б задати
рівні структури самим стилям, але це вже зміна офіційного шаблону,
яку перевірка на етапі 5 може відхилити.

Списків рисунків і таблиць процедура не вимагає: у таблиці 1.1
структура роботи їх немає. Якщо керівник попросить додати, поля
готові до цього, бо підписи містять SEQ:

    TOC \\h \\z \\c "Рисунок"
    TOC \\h \\z \\c "Таблиця"

=== ЯК ОНОВИТИ ВСІ ПОЛЯ ===

Після відкриття файлу зміст порожній, а замість номерів підписів стоїть
одиниця: поля ще не обчислені. Оновити так:

    1. Ctrl+A - виділити весь документ
    2. F9 - оновити поля
    3. На діалог «Оновити таблицю» відповісти «Оновити цілком»
       (Update entire table), а не «Оновити лише номери сторінок»

На macOS F9 може бути перехоплена системою; тоді використати
Fn+F9 або меню правою кнопкою на полі -> «Оновити поле».

Оновлення повторити після кожної зміни структури: додавання розділу,
таблиці або рисунка.
"""



# ------------------------------------------------------- текст з розміткою

BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.S)
CODE_RE = re.compile(r"`([^`]+)`", re.S)
TOKEN_RE = re.compile(r"(\*\*.+?\*\*|`[^`]+`)", re.S)


def add_rich_text(paragraph, text: str):
    """Розмітка чернетки в справжні прогони Word.

    `**текст**` стає жирним прогоном, `` `код` `` — прогоном гарнітурою
    Consolas без зворотних лапок. Решта символів розмітки в документ
    не потрапляє: чернетки пишуться в markdown, готовий файл markdown
    не містить.
    """
    for part in TOKEN_RE.split(text):
        if not part:
            continue
        bold = BOLD_RE.fullmatch(part)
        code = CODE_RE.fullmatch(part)
        if bold:
            paragraph.add_run(bold.group(1)).bold = True
        elif code:
            run = paragraph.add_run(code.group(1))
            run.font.name = "Consolas"
        else:
            paragraph.add_run(part)
    return paragraph


def add_paragraph(document, text: str, style: str):
    paragraph = document.add_paragraph(style=style)
    add_rich_text(paragraph, text)
    return paragraph


DOI_RE = re.compile(r"DOI:\s*(10\.\S+?)(?=[\s,;]|$)")
ARXIV_RE = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5})")


def add_hyperlink(paragraph, url: str, text: str):
    """Справжнє поле посилання: w:hyperlink з зовнішнім відношенням."""
    part = paragraph.part
    r_id = part.relate_to(url, RT.HYPERLINK, is_external=True)
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    props = OxmlElement("w:rPr")
    style = OxmlElement("w:rStyle")
    style.set(qn("w:val"), "Hyperlink")
    props.append(style)
    run.append(props)
    node = OxmlElement("w:t")
    node.text = text
    run.append(node)
    link.append(run)
    paragraph._p.append(link)


def add_reference(document, text: str):
    """Позиція списку джерел; DOI та arXiv стають клікабельними."""
    paragraph = document.add_paragraph(style="References")
    match = DOI_RE.search(text) or ARXIV_RE.search(text)
    if not match:
        paragraph.add_run(text)
        return paragraph, False
    paragraph.add_run(text[: match.start()])
    identifier = match.group(1)
    url = (
        f"https://doi.org/{identifier}"
        if match.re is DOI_RE
        else f"https://arxiv.org/abs/{identifier}"
    )
    prefix = "DOI: " if match.re is DOI_RE else "arXiv: "
    paragraph.add_run(prefix)
    add_hyperlink(paragraph, url, identifier)
    paragraph.add_run(text[match.end():])
    return paragraph, True


# ------------------------------------------------- титульна і завдання

# Значення зі шаблону, підсвічені жовтим, і чим їх замінює скрипт.
# Підсвічення зберігається: студент має заповнити місце і зняти колір.
FRONT_REPLACEMENTS = {
    "Тема роботи": "Прогнозування залученості користувачів до рекламних "
                   "оголошень на основі характеристик реклами",
    "Володимира ШИМАНСЬКОГО": "ІМ'Я ПРІЗВИЩЕ ЗДОБУВАЧА",
    "Шиманському Володимиру Михайловичу": "ПРІЗВИЩЕ ІМ'Я ПО БАТЬКОВІ ЗДОБУВАЧА",
}


def copy_front_matter(document, template_path: Path) -> int:
    """Скопіювати бланки титульної й завдання з шаблону.

    Бланки лежать усередині елемента `sdt` на початку тіла шаблону,
    тому в звичайному переліку абзаців їх не видно. Копіюються як є,
    разом з таблицями підписів і жовтим підсвіченням; підставляються
    лише значення-зразки попереднього студента.
    """
    source = docx.Document(str(template_path))
    body = source.element.body
    sdt = body.find(qn("w:sdt"))
    if sdt is None:
        return 0
    content = sdt.find(qn("w:sdtContent"))
    copied = replaced = 0
    for element in list(content):
        clone = copy.deepcopy(element)
        # Заміна лише в підсвічених жовтим прогонах. Без цієї умови
        # підстановка зачіпає збіги в звичайному тексті бланка:
        # у першій версії скрипта так постраждав рік у рядку дати.
        for run in clone.iter(qn("w:r")):
            props = run.find(qn("w:rPr"))
            if props is None or props.find(qn("w:highlight")) is None:
                continue
            for node in run.iter(qn("w:t")):
                key = (node.text or "").strip()
                if key in FRONT_REPLACEMENTS:
                    node.text = FRONT_REPLACEMENTS[key]
                    replaced += 1
        document.element.body.append(clone)
        copied += 1
    return copied, replaced


# ---------------------------------------------------------------- шаблон

def repack_template() -> Path:
    """Перепакувати .dotm у .docx, придатний для python-docx."""
    SCRATCH.parent.mkdir(parents=True, exist_ok=True)
    macro_parts = ("word/vbaProject.bin", "word/vbaData.xml")
    with zipfile.ZipFile(TEMPLATE) as src, zipfile.ZipFile(
        SCRATCH, "w", zipfile.ZIP_DEFLATED
    ) as dst:
        for item in src.namelist():
            if item in macro_parts:
                continue
            data = src.read(item)
            if item == "[Content_Types].xml":
                text = data.decode("utf-8")
                text = text.replace(
                    "application/vnd.ms-word.template.macroEnabledTemplate.main+xml",
                    "application/vnd.openxmlformats-officedocument"
                    ".wordprocessingml.document.main+xml",
                )
                data = text.encode("utf-8")
            dst.writestr(item, data)
    return SCRATCH


def clear_body(document) -> None:
    """Видалити приклад зі шаблону, зберігши параметри сторінки."""
    body = document.element.body
    sect = body.find(qn("w:sectPr"))
    for child in list(body):
        if child is not sect:
            body.remove(child)


def check_styles(document) -> dict:
    """Звірити наявні в шаблоні стилі з тими, що потрібні скрипту."""
    present = {s.name for s in document.styles}
    report = {"present": [], "missing": []}
    for name in REQUIRED_STYLES:
        (report["present"] if name in present else report["missing"]).append(name)
    return report


def outline_levels(document) -> dict:
    """Рівень структури для кожного стилю заголовка, з урахуванням basedOn.

    Потрібно тому, що поле TOC із перемикачем \\o збирає абзаци за рівнем
    структури, а не за назвою стилю. Стиль без рівня в зміст не потрапить.
    """
    styles = {s.style_id: s for s in document.styles}
    by_name = {s.name: s for s in document.styles}

    def level(style):
        seen = set()
        while style is not None and style.style_id not in seen:
            seen.add(style.style_id)
            ppr = style.element.find(qn("w:pPr"))
            if ppr is not None:
                node = ppr.find(qn("w:outlineLvl"))
                if node is not None:
                    return int(node.get(qn("w:val"))), style.name
            based = style.element.find(qn("w:basedOn"))
            style = styles.get(based.get(qn("w:val"))) if based is not None else None
        return None, None

    return {
        name: level(by_name[name])
        for name in ("Header_Level_1", "Header_Level_2", "Header_Level_3")
        if name in by_name
    }


# ------------------------------------------------------------------ поля

def add_field(paragraph, instruction: str, placeholder: str) -> None:
    """Справжнє поле Word: begin -> instrText -> separate -> текст -> end."""
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    run._r.append(begin)

    run = paragraph.add_run()
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    run._r.append(instr)

    run = paragraph.add_run()
    sep = OxmlElement("w:fldChar")
    sep.set(qn("w:fldCharType"), "separate")
    run._r.append(sep)

    paragraph.add_run(placeholder)

    run = paragraph.add_run()
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(end)


def add_seq(paragraph, identifier: str, reset: bool = False) -> None:
    """Нумераційне поле SEQ. Дає автоматичну нумерацію підписів.

    Перемикач \\r 1 скидає лічильник на початку кожного розділу, бо
    нумерація в роботі порозділова: «Рис. 3.1», а не «Рис. 12».
    """
    switches = " \\* ARABIC" + (" \\r 1" if reset else "")
    add_field(paragraph, f" SEQ {identifier}{switches} ", "1")


# --------------------------------------------------------------- розбір

MARKER_RE = re.compile(r"^\[([^\]]+)\]\s*(.*)$")


def read_blocks(path: Path, stop: str | None):
    """Абзаци чернетки як (позначка, текст), від Header_Level_1 до stop."""
    lines = path.read_text(encoding="utf-8").split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("[Header_Level_1]"))
    end = len(lines)
    if stop:
        for i, line in enumerate(lines):
            if line.startswith(stop):
                end = i
                break

    blocks = []
    marker = None
    buffer: list[str] = []

    def flush():
        if marker and buffer:
            blocks.append((marker, " ".join(buffer).strip()))

    for line in lines[start:end]:
        match = MARKER_RE.match(line)
        if match and (match.group(1) in MARKER_STYLE or match.group(1) == "ТАБЛИЦЯ"):
            flush()
            marker, buffer = match.group(1), [match.group(2)]
        elif not line.strip():
            flush()
            marker, buffer = None, []
        elif marker:
            buffer.append(line.strip())
    flush()
    return blocks


def parse_table_row(text: str) -> list[str]:
    text = re.sub(r"^Стовпці:\s*", "", text)
    return [cell.strip() for cell in text.split("|")]


# --------------------------------------------------------- перевірка

ARTEFACTS = {
    "жирний markdown **": re.compile(r"\*\*"),
    # Рівно дві риски з кожного боку. Довші послідовності — це лінії
    # для рукописного заповнення в бланку завдання, а не markdown.
    "підкреслення __текст__": re.compile(
        r"(?<!_)__(?!_)[^_\n]{1,80}(?<!_)__(?!_)"
    ),
    "зворотні лапки `": re.compile(r"`"),
    "заголовок markdown #": re.compile(r"^#{1,6}\s"),
    "позначка стилю [Marker]": re.compile(
        r"\[(?:Main_Text|Header_Level_\d|Marked_List|Numbered_List|"
        r"Table_Title|Figure_Title|ТАБЛИЦЯ)\]"
    ),
}


def full_text(element) -> str:
    """Увесь текст елемента, включно з вкладеними `sdt`.

    Властивість `.text` у python-docx бере лише прогони, що є прямими
    нащадками абзацу. Текст усередині елементів керування вмістом вона
    пропускає, і на бланку завдання це давало хибне «20р.» замість
    «2025р.». Перевірка має бачити те саме, що побачить Word.
    """
    return "".join(node.text or "" for node in element.iter(qn("w:t")))


def verify(path: Path) -> dict:
    """Пошук слідів markdown у готовому файлі.

    Перевіряються і абзаци, і комірки таблиць: розмітка найлегше
    лишається саме в таблицях, бо їх збирає окрема гілка коду.
    """
    document = docx.Document(str(path))
    places = [("абзац", full_text(p._p)) for p in document.paragraphs]
    for i, table in enumerate(document.tables):
        for row in table.rows:
            for cell in row.cells:
                places.append((f"таблиця {i + 1}", full_text(cell._tc)))

    found = {}
    for name, pattern in ARTEFACTS.items():
        hits = [text for kind, text in places if pattern.search(text)]
        found[name] = hits

    xml = document.element.body.xml
    return {
        "artefacts": found,
        "checked": len(places),
        "hyperlinks": xml.count("<w:hyperlink"),
        "fields_seq": xml.count("SEQ "),
        "fields_toc": xml.count("TOC "),
        "drawings": xml.count("<w:drawing>"),
    }


# --------------------------------------------------------------- збірка

def add_placeholder_section(document, title: str, lines: list[str]) -> None:
    document.add_paragraph(title, style="Header_Level_1")
    for line in lines:
        document.add_paragraph(line, style="Main_Text")
    document.add_page_break()


def add_toc(document) -> None:
    """Зміст полем TOC.

    Перемикач \\o "1-3" збирає абзаци за рівнем структури. У шаблоні
    рівень заданий лише для Header_Level_1, тому рівні 2 і 3 додаються
    перемикачем \\t, який відображає назви стилів у рівні змісту.
    Без \\t підрозділи в зміст не потрапили б.
    """
    document.add_paragraph("ЗМІСТ", style="Header_Level_1")
    paragraph = document.add_paragraph(style="Main_Text")
    add_field(
        paragraph,
        ' TOC \\o "1-3" \\t "Header_Level_2,2,Header_Level_3,3" \\h \\z \\u ',
        "Зміст порожній. Оновіть поле: Ctrl+A, потім F9.",
    )
    document.add_page_break()


def add_table(document, rows: list[list[str]]) -> None:
    width = max(len(r) for r in rows)
    table = document.add_table(rows=len(rows), cols=width)
    table.style = "Table Grid"
    for i, row in enumerate(rows):
        for j in range(width):
            cell = table.cell(i, j)
            cell.text = ""
            paragraph = cell.paragraphs[0]
            paragraph.style = document.styles["Main_Text"]
            paragraph.paragraph_format.first_line_indent = Cm(0)
            add_rich_text(paragraph, row[j] if j < len(row) else "")
    return table


def add_caption(document, style: str, kind: str, number: str, title: str,
                counters: dict) -> None:
    """Підпис із полем SEQ замість набраного вручну номера."""
    chapter, index = number.split(".")
    paragraph = document.add_paragraph(style=style)
    paragraph.add_run(f"{kind} {chapter}.")
    reset = counters.get(kind) != chapter
    counters[kind] = chapter
    add_seq(paragraph, "Рисунок" if kind == "Рис." else "Таблиця", reset=reset)
    paragraph.add_run(". ")
    add_rich_text(paragraph, title)


def add_figure(document, number: str, title: str, counters: dict,
               missing: list) -> None:
    filename = FIGURE_FILES.get(number)
    path = FIGURES / filename if filename else None
    if path and path.exists():
        picture = document.add_paragraph(style="Main_Text")
        picture.alignment = WD_ALIGN_PARAGRAPH.CENTER
        picture.paragraph_format.first_line_indent = Cm(0)
        picture.add_run().add_picture(str(path), width=Cm(16))
    else:
        missing.append((number, filename))
        document.add_paragraph(
            f"[{PLACEHOLDER}: рисунок {number}, файл {filename or 'не вказано'}]",
            style="Main_Text",
        )
    add_caption(document, "Figure_Title", "Рис.", number, title, counters)


def add_references(document) -> None:
    document.add_paragraph("СПИСОК ВИКОРИСТАНИХ ДЖЕРЕЛ", style="Header_Level_1")
    text = Path("docs/references.md").read_text(encoding="utf-8")
    body = text.split("1. Berger", 1)[1].split("\n\n---", 1)[0]
    entries = re.split(r"\n(?=\d{1,2}\. )", "1. Berger" + body)
    linked = 0
    for entry in entries:
        cleaned = re.sub(r"^\d{1,2}\.\s*", "", " ".join(entry.split()))
        _, has_link = add_reference(document, cleaned)
        linked += int(has_link)
    document.add_page_break()
    return len(entries), linked


def add_appendices(document) -> None:
    appendices = [
        ("Додаток А", "Рубрика анотації копірайтингових характеристик",
         "annotations/rubric_v2.md"),
        ("Додаток Б", "Інструкція анотатора", "annotations/prompt_v1.txt"),
        ("Додаток В", "Код побудови ознак і перевірки на витік",
         "src/features/, src/build_features.py"),
        ("Додаток Г", "Код експериментів і обчислення метрик",
         "src/experiment.py, src/compare.py, src/eval/"),
        ("Додаток Д", "Повні таблиці результатів", "reports/tables/"),
    ]
    for title, description, source in appendices:
        document.add_paragraph(title, style="Header_Level_1")
        document.add_paragraph(description, style="Main_Text")
        document.add_paragraph(
            f"[{PLACEHOLDER}: вставити вміст з {source}]", style="Main_Text"
        )
        document.add_page_break()


def build() -> dict:
    repack_template()
    document = docx.Document(str(SCRATCH))

    styles = check_styles(document)
    if styles["missing"]:
        raise SystemExit(
            "У шаблоні немає стилів: " + ", ".join(styles["missing"])
            + ". Збірку припинено: підставляти схожий стиль не можна."
        )
    levels = outline_levels(document)
    clear_body(document)

    n_front, n_replaced = copy_front_matter(document, SCRATCH)
    document.add_page_break()
    add_placeholder_section(document, "АНОТАЦІЯ", [
        f"[{PLACEHOLDER}] 1 сторінка. Склад полів — docs/thesis_structure.md",
        "розділ 10. Рядок про обсяг заповнюється після верстки.",
    ])
    add_placeholder_section(document, "ABSTRACT", [
        f"[{PLACEHOLDER}] 10–15 сторінок: захист англійською мовою.",
        "Не переклад анотації, а самостійний переказ роботи.",
        "Терміни — docs/glossary.md.",
    ])
    add_toc(document)

    counters: dict = {}
    missing_figures: list = []
    stats = {"paragraphs": 0, "tables": 0, "figures": 0}

    for path, stop in PARTS:
        blocks = read_blocks(Path(path), stop)
        pending_rows: list[list[str]] = []
        for marker, text in blocks:
            if marker == "ТАБЛИЦЯ":
                pending_rows.append(parse_table_row(text))
                continue
            if pending_rows:
                add_table(document, pending_rows)
                pending_rows = []
                stats["tables"] += 1
            if marker == "Table_Title":
                match = re.match(r"Таблиця (\d+\.\d+)\.\s*(.*)", text)
                if match:
                    add_caption(document, "Table_Title", "Таблиця",
                                match.group(1), match.group(2), counters)
                else:
                    add_paragraph(document, text, "Table_Title")
            elif marker == "Figure_Title":
                match = re.match(r"Рис\. (\d+\.\d+)\.\s*(.*)", text)
                if match:
                    add_figure(document, match.group(1), match.group(2),
                               counters, missing_figures)
                    stats["figures"] += 1
                else:
                    add_paragraph(document, text, "Figure_Title")
            else:
                add_paragraph(document, text, MARKER_STYLE[marker])
                stats["paragraphs"] += 1
        if pending_rows:
            add_table(document, pending_rows)
            stats["tables"] += 1
        document.add_page_break()

    n_refs, n_linked = add_references(document)
    add_appendices(document)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(OUT))
    return {
        "styles": styles,
        "levels": levels,
        "stats": stats,
        "missing_figures": missing_figures,
        "references": n_refs,
        "linked": n_linked,
        "front_elements": n_front,
        "front_replaced": n_replaced,
    }


def main() -> None:
    print("Стилі шаблону")
    repack_template()
    probe = docx.Document(str(SCRATCH))
    all_styles = sorted(s.name for s in probe.styles)
    print(f"Усього стилів у шаблоні: {len(all_styles)}")
    report = check_styles(probe)
    for name in REQUIRED_STYLES:
        mark = "є" if name in report["present"] else "НЕМАЄ"
        print(f"  {name:16s} {mark}")
    if report["missing"]:
        print("\nЗбірку припинено: у шаблоні немає потрібних стилів.")
        raise SystemExit(1)

    print("\nРівні структури заголовків")
    for name, (level, source) in outline_levels(probe).items():
        if level is None:
            print(f"  {name:16s} рівня НЕ ЗАДАНО")
        else:
            print(f"  {name:16s} рівень {level} (успадковано від «{source}»)")

    result = build()
    stats = result["stats"]
    print("\nЗібрано")
    print(f"  файл: {OUT}")
    print(f"  абзаців тексту: {stats['paragraphs']}")
    print(f"  таблиць: {stats['tables']}")
    print(f"  рисунків вставлено: {stats['figures'] - len(result['missing_figures'])}"
          f" з {stats['figures']}")
    print(f"  джерел у списку: {result['references']}, "
          f"з клікабельним DOI або arXiv: {result['linked']}")
    print(f"  елементів бланків титульної й завдання: {result['front_elements']}, "
          f"підставлено значень: {result['front_replaced']}")
    if result["missing_figures"]:
        print("  НЕ ЗНАЙДЕНО файлів рисунків:")
        for number, filename in result["missing_figures"]:
            print(f"    {number}: {filename}")

    report = verify(OUT)
    print("\nПеревірка на артефакти")
    print(f"  перевірено місць (абзаци + комірки): {report['checked']}")
    total = 0
    for name, hits in report["artefacts"].items():
        total += len(hits)
        mark = "чисто" if not hits else f"ЗНАЙДЕНО {len(hits)}"
        print(f"  {name:26s} {mark}")
        for text in hits[:3]:
            print(f"      {text[:70]}")
    print(f"  разом артефактів: {total}")
    print(f"  полів HYPERLINK: {report['hyperlinks']}")
    print(f"  полів SEQ: {report['fields_seq']}, полів TOC: {report['fields_toc']}")
    print(f"  зображень: {report['drawings']}")

    print(MANUAL_STEPS)


if __name__ == "__main__":
    main()
