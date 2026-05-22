<div dir="rtl" lang="he">

# ספר פרויקט – RouteMind

## מערכת אופטימיזציה למסלולי שילוח עם חלונות זמן ועדיפויות

---

**שם הפרויקט:** RouteMind – Route Optimization System

**סוג הפרויקט:** פרויקט גמר בהנדסת תוכנה

**תאריך הגשה:** 2026

---

## תוכן עניינים

1. [תקציר מנהלים](#1-תקציר-מנהלים)
2. [מבוא](#2-מבוא)
3. [רקע תיאורטי](#3-רקע-תיאורטי)
4. [הגדרת הבעיה](#4-הגדרת-הבעיה)
5. [דרישות המערכת](#5-דרישות-המערכת)
6. [ארכיטקטורת המערכת](#6-ארכיטקטורת-המערכת)
7. [תכנון מפורט](#7-תכנון-מפורט)
8. [מימוש](#8-מימוש)
9. [ממשק משתמש](#9-ממשק-משתמש)
10. [בדיקות](#10-בדיקות)
11. [אבטחת מידע](#11-אבטחת-מידע)
12. [פריסה והרצה](#12-פריסה-והרצה)
13. [סיכום ומסקנות](#13-סיכום-ומסקנות)
14. [נספחים](#14-נספחים)

---

## 1. תקציר מנהלים

**RouteMind** היא מערכת אופטימיזציה למסלולי שילוח המבוססת על אלגוריתמים הוריסטיים לפתרון בעיית ניתוב רכבים עם חלונות זמן (Vehicle Routing Problem with Time Windows – VRPTW). המערכת מאפשרת למשתמשים להגדיר נקודות עצירה עם מגבלות זמן ועדיפויות, ולקבל מסלול אופטימלי תוך שקלול מרחק, זמני המתנה, איחורים ועומסי משמרת.

המערכת מורכבת משרת API מבוסס FastAPI עם מנוע אופטימיזציה הכולל בנייה חמדנית (Greedy) של מסלול התחלתי ושיפורו באמצעות אלגוריתמי חיפוש מקומי (2-opt, Swap, Relocate). כמו כן, המערכת כוללת ממשק דפדפן אינטראקטיבי עם תמיכה במפות, ג'אוקודינג, ניהול תרחישים, והיסטוריית אופטימיזציות.

הטכנולוגיות העיקריות: Python 3.11+, FastAPI, Pydantic v2, SQLite, Uvicorn, HTML/CSS/JavaScript (Vanilla), Leaflet.js למפות, ו-Docker לפריסה.

---

## 2. מבוא

### 2.1 הצגת הנושא

בעיית ניתוב רכבים (VRP) היא אחת הבעיות המרכזיות במחקר ביצועים ובלוגיסטיקה. במקרה הכללי, מטרת הבעיה היא למצוא את סדר הביקורים האופטימלי בקבוצת נקודות (לקוחות, עצירות) כך שעלות הנסיעה הכוללת תהיה מינימלית, בכפוף למגבלות שונות.

הווריאנט VRPTW (Vehicle Routing Problem with Time Windows) מוסיף מגבלות חלונות זמן – לכל נקודת עצירה יש זמן מוקדם ביותר וזמן מאוחר ביותר שבהם ניתן להגיע אליה. תוספת זו הופכת את הבעיה למורכבת יותר באופן משמעותי, ודורשת איזון בין קיצור מרחקים לבין עמידה במגבלות הזמן.

### 2.2 מוטיבציה

חברות שילוח, שירותי משלוחים, וארגוני שירות שדה מתמודדים יומיומית עם אתגר תכנון מסלולים יעיל. תכנון לא אופטימלי מוביל ל:
- עלויות דלק מיותרות
- איחורים ללקוחות
- חריגות ממשמרות עבודה
- ירידה בשביעות רצון הלקוחות

**RouteMind** נועדה לספק כלי נגיש, מהיר וגמיש לאופטימיזציה של מסלולים, עם דגש על:
- תמיכה בחלונות זמן לכל עצירה
- מערכת עדיפויות (1–5) להבחנה בין עצירות קריטיות לפחות דחופות
- אכיפת מגבלת משמרת (shift time) למניעת חריגה
- משקלות ניתנים לכיוונון לכל רכיב עלות
- ממשק ויזואלי אינטואיטיבי עם מפות

### 2.3 מטרות הפרויקט

1. לפתח מנוע אופטימיזציה המיישם מספר אלגוריתמי חיפוש מקומי לשיפור מסלולים
2. לבנות שרת API מאובטח עם מערכת הרשאות ואימות מבוססת JWT
3. לספק ממשק דפדפן אינטראקטיבי המאפשר הזנת נתונים, הרצת אופטימיזציה, וויזואליזציה של התוצאות על מפה
4. לאפשר ניהול תרחישים (שמירה, עדכון, מחיקה) והיסטוריית הרצות
5. לתמוך בהזנת כתובות עם ג'אוקודינג וחישוב מרחקים גיאוגרפיים (Haversine)
6. לספק מערכת מוכנה לפריסה באמצעות Docker

---

## 3. רקע תיאורטי

### 3.1 בעיית ניתוב רכבים (VRP)

בעיית ניתוב רכבים היא בעיית אופטימיזציה קומבינטורית NP-קשה. בגרסה הבסיסית, נתון צי רכבים היוצא ממחסן (depot) ומבקר בקבוצת לקוחות. המטרה: למצוא את קבוצת המסלולים שממזערת את העלות הכוללת.

### 3.2 VRPTW – חלונות זמן

בווריאנט VRPTW, לכל לקוח i מוגדר חלון זמן [eᵢ, lᵢ] כאשר:
- eᵢ = הזמן המוקדם ביותר לתחילת שירות
- lᵢ = הזמן המאוחר ביותר לתחילת שירות

אם הרכב מגיע לפני eᵢ, הוא ממתין. אם מגיע אחרי lᵢ, נוצר איחור (lateness) שמהווה עלות/קנס.

### 3.3 גישות פתרון

#### 3.3.1 בנייה חמדנית (Greedy Construction)

אלגוריתם הבנייה מתחיל ממחסן ריק ובכל שלב בוחר את העצירה הבאה שמביאה למינימום עלות שולית, תוך העדפת מהלכים שמשמרים ישימות (feasibility). הבחירה מבוססת על שקלול של:
- מרחק נסיעה
- זמן המתנה צפוי
- איחור צפוי (משוקלל לפי עדיפות)

#### 3.3.2 חיפוש מקומי – 2-opt

אלגוריתם 2-opt מבצע באופן איטרטיבי היפוך של תת-מקטע במסלול. אם ההיפוך מוריד את העלות הכוללת (או משפר ישימות), המהלך מתקבל. האלגוריתם ממשיך עד שמתקיים אחד מתנאי העצירה:
- הגעה למספר איטרציות מקסימלי
- הגעה למגבלת איטרציות ללא שיפור
- חריגה ממגבלת זמן ריצה

#### 3.3.3 חיפוש מקומי – Swap

אלגוריתם Swap בוחן כל זוג עצירות במסלול ומחליף ביניהן. אם ההחלפה מורידה עלות – היא מתקבלת. משתמש באותם תנאי עצירה כמו 2-opt.

#### 3.3.4 חיפוש מקומי – Relocate

אלגוריתם Relocate מנסה להזיז כל עצירה לכל מיקום אחר במסלול. אם ההזזה מורידה עלות – היא מתקבלת. משתמש באותם תנאי עצירה כמו 2-opt.

### 3.4 פונקציית העלות

פונקציית העלות הכוללת מוגדרת כ:

```
total_cost = w_dist × total_travel_distance
           + w_wait × total_wait_time
           + w_late × priority_adjusted_lateness
           + w_shift × shift_overrun
```

כאשר:
- `w_dist` – משקל מרחק (ברירת מחדל: 1.0)
- `w_wait` – משקל זמן המתנה (ברירת מחדל: 1.0)
- `w_late` – משקל איחור (ברירת מחדל: 2.0)
- `w_priority` – משקל עדיפות (ברירת מחדל: 2.0)
- `w_shift` – משקל חריגת משמרת (ברירת מחדל: 4.0)

#### מקדם עדיפות (Priority Factor)

```
priority_factor(p) = 1 + (p − 1) × 0.5 × max(w_priority, 0)
```

עצירה בעדיפות 1 מקבלת מקדם 1.0 (ללא קנס נוסף). כל רמת עדיפות נוספת מוסיפה `0.5 × w_priority` למכפיל. לדוגמה, עצירה בעדיפות 5 עם `w_priority=2.0` נושאת מקדם של 5.0.

#### חישוב איחור משוקלל

```
priority_adjusted_lateness = Σᵢ lateness_i × priority_factor(priority_i)
```

### 3.5 ישימות מסלול

מסלול נחשב ישים (feasible) אם ורק אם זמן הסיום הכולל (כולל חזרה למחסן) אינו חורג מזמן המשמרת המקסימלי. כלומר:

```
feasible = (finish_time ≤ max_shift_time)
```

### 3.6 חישוב מרחקים

המערכת תומכת בשני מצבי חישוב מרחק:

1. **מרחק אוקלידי** – לתרחישים מופשטים ללא קואורדינטות גיאוגרפיות
2. **Haversine + זמן נסיעה** – כאשר נקודות מוגדרות בקואורדינטות lat/lng, המערכת מחשבת מרחק גיאוגרפי (Great-circle) וממירה לזמן נסיעה בדקות לפי מהירות ממוצעת:
   - נהיגה: 50 קמ"ש
   - הליכה: 5 קמ"ש

---

## 4. הגדרת הבעיה

### 4.1 הגדרה פורמלית

**נתון:**
- מחסן (depot) D עם קואורדינטות (x, y) או (lat, lng)
- קבוצת עצירות S = {s₁, s₂, ..., sₙ} כאשר לכל עצירה sᵢ:
  - מיקום: (xᵢ, yᵢ) או (latᵢ, lngᵢ)
  - חלון זמן: [window_startᵢ, window_endᵢ]
  - זמן שירות: service_timeᵢ ≥ 0
  - עדיפות: priorityᵢ ∈ {1, 2, 3, 4, 5}
- זמן משמרת מקסימלי: max_shift_time > 0
- וקטור משקלות: W = (w_dist, w_wait, w_late, w_priority, w_shift)

**למצוא:**
- סדר ביקור π של כל העצירות ב-S, כך שפונקציית העלות הכוללת מינימלית, בכפוף למגבלות הישימות.

### 4.2 מגבלות

1. כל עצירה נבקרת בדיוק פעם אחת
2. המסלול מתחיל ומסתיים במחסן
3. חריגה מזמן משמרת אפשרית אך כרוכה בקנס גבוה
4. איחור לחלון זמן אפשרי אך כרוך בקנס (משוקלל לפי עדיפות)

### 4.3 תרחיש שימוש טיפוסי

1. משתמש נרשם למערכת ומתחבר
2. מגדיר מחסן (לפי כתובת או קואורדינטות)
3. מוסיף עצירות עם פרטי מיקום, חלונות זמן, ועדיפויות
4. בוחר אלגוריתם אופטימיזציה (2-opt / Swap / Relocate)
5. מריץ אופטימיזציה ומקבל תוצאות: סדר מסלול, עלויות, ויזואליזציה על מפה
6. שומר תרחיש לשימוש עתידי
7. צופה בהיסטוריית הרצות קודמות

---

## 5. דרישות המערכת

### 5.1 דרישות פונקציונליות

| מס' | דרישה | תיאור |
|-----|--------|--------|
| FR-1 | רישום משתמשים | המערכת תאפשר רישום עם שם משתמש, דוא"ל וסיסמה |
| FR-2 | התחברות | המערכת תאפשר התחברות לפי שם משתמש או דוא"ל |
| FR-3 | אימות JWT | כל פעולה מוגנת דורשת טוקן Bearer תקף |
| FR-4 | הגדרת מחסן | המשתמש יוכל להגדיר מחסן לפי קואורדינטות או כתובת |
| FR-5 | הגדרת עצירות | המשתמש יוכל להוסיף עצירות עם חלונות זמן ועדיפויות |
| FR-6 | אופטימיזציה | המערכת תחשב מסלול אופטימלי באמצעות האלגוריתם הנבחר |
| FR-7 | בחירת אלגוריתם | המשתמש יוכל לבחור בין 2-opt, Swap ו-Relocate |
| FR-8 | ויזואליזציה | המערכת תציג את המסלול המותאם על מפה אינטראקטיבית |
| FR-9 | ניהול תרחישים | שמירה, טעינה, עדכון ומחיקה של תרחישי מסלול |
| FR-10 | היסטוריה | המערכת תשמור ותציג היסטוריית הרצות אופטימיזציה |
| FR-11 | ג'אוקודינג | המערכת תתמוך בהמרת כתובות לקואורדינטות |
| FR-12 | מצבי נסיעה | תמיכה במצב נהיגה ומצב הליכה |
| FR-13 | ניתוב גיאוגרפי | הצגת מסלול מציאותי על מפה (לא קו ישר) |
| FR-14 | ערכת נושא | תמיכה במצב בהיר ומצב כהה |
| FR-15 | בדיקת תקינות | נקודת קצה /health לניטור תקינות השרת |

### 5.2 דרישות לא-פונקציונליות

| מס' | דרישה | תיאור |
|-----|--------|--------|
| NFR-1 | ביצועים | אופטימיזציה של עד 50 עצירות בפחות מ-10 שניות |
| NFR-2 | אבטחה | סיסמאות מוצפנות ב-scrypt; טוקנים חתומים ב-HS256 |
| NFR-3 | הגבלת קצב | מנגנון Rate Limiting למניעת שימוש לרעה |
| NFR-4 | ולידציה | כל הקלטים עוברים ולידציה מלאה עם Pydantic |
| NFR-5 | זמינות | ניתן להריץ כ-container עצמאי עם Docker |
| NFR-6 | תאימות | תומך ב-Python 3.11 ומעלה |
| NFR-7 | רספונסיביות | ממשק המשתמש מותאם למסכים שונים |
| NFR-8 | נגישות | תמיכה ב-CORS לגישה מאפליקציות חיצוניות |

### 5.3 מגבלות עיצוב

- בסיס הנתונים: SQLite (ללא תלות בשרת חיצוני)
- אין תלות בשירותים חיצוניים לליבת האופטימיזציה
- ממשק משתמש: Vanilla JavaScript ללא frameworks
- הפריסה: container יחיד עצמאי

---

## 6. ארכיטקטורת המערכת

### 6.1 תרשים ארכיטקטורה כללי

```
┌─────────────────────────────────────────────────────────────┐
│                     לקוח (דפדפן)                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  HTML/CSS/JS │ Leaflet.js │ Geocoding │ Theme Engine    ││
│  └─────────────────────────────────────────────────────────┘│
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP/REST (JSON)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     שרת FastAPI                              │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌───────────┐ │
│  │   Auth   │  │ Optimize │  │ Scenarios │  │  History   │ │
│  │  Module  │  │  Engine  │  │  Storage  │  │  Storage   │ │
│  └──────────┘  └──────────┘  └───────────┘  └───────────┘ │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌───────────┐ │
│  │   Rate   │  │ Distance │  │  Geocode  │  │   Road    │ │
│  │ Limiter  │  │  Matrix  │  │  Search   │  │  Router   │ │
│  └──────────┘  └──────────┘  └───────────┘  └───────────┘ │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     SQLite Database                           │
│  ┌──────────────┐  ┌───────────────┐  ┌─────────────────┐  │
│  │    users     │  │saved_scenarios│  │optimization_runs│  │
│  └──────────────┘  └───────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 מבנה תיקיות הפרויקט

```
RouteMind-Project/
├── routemind/
│   ├── app/
│   │   ├── main.py          # אפליקציית FastAPI, נתיבים, middleware
│   │   ├── models.py        # מודלי Pydantic לבקשות ותגובות
│   │   ├── auth.py          # אימות: רישום, התחברות, JWT, הצפנת סיסמאות
│   │   ├── storage.py       # שמירת תרחישים והיסטוריית אופטימיזציות (SQLite)
│   │   ├── optimizer.py     # בנייה חמדנית + אלגוריתמי 2-opt / swap / relocate
│   │   ├── evaluator.py     # הערכת עלות מסלול
│   │   ├── rate_limit.py    # מגביל קצב בזיכרון (sliding window)
│   │   └── utils.py         # חישובי מרחק, בניית מטריצת מרחקים
│   ├── static/
│   │   └── index.html       # ממשק משתמש בדפדפן (SPA)
│   ├── tests/
│   │   └── test_routemind.py # בדיקות יחידה ואינטגרציה
│   ├── auth_store/          # בסיס נתונים SQLite לאימות (נוצר אוטומטית)
│   └── requirements.txt     # תלויות Python
├── Dockerfile               # תמונת Docker
├── docker-compose.yml       # תצורת Docker Compose
├── README.md                # תיעוד הפרויקט באנגלית
└── .gitignore
```

### 6.3 שכבות המערכת

המערכת בנויה בארכיטקטורת שכבות (Layered Architecture):

1. **שכבת הצגה (Presentation Layer)** – ממשק HTML/CSS/JS סטטי המוגש ישירות מהשרת
2. **שכבת API (API Layer)** – נקודות קצה REST מבוססות FastAPI
3. **שכבת לוגיקה עסקית (Business Logic Layer)** – מנוע אופטימיזציה, מעריך עלות, חישובי מרחק
4. **שכבת נתונים (Data Layer)** – SQLite לאחסון משתמשים, תרחישים, והיסטוריה

---

## 7. תכנון מפורט

### 7.1 מודל הנתונים

#### 7.1.1 טבלת משתמשים (users)

| שדה | סוג | תיאור |
|-----|------|--------|
| id | INTEGER PRIMARY KEY | מזהה ייחודי |
| username | TEXT UNIQUE NOT NULL | שם משתמש (3–32 תווים, אותיות/מספרים/קו_תחתון/מקף) |
| email | TEXT UNIQUE NOT NULL | כתובת דוא"ל מנורמלת |
| password_hash | TEXT NOT NULL | hash בפורמט scrypt |
| created_at | TEXT NOT NULL | חותמת זמן ISO 8601 |

#### 7.1.2 טבלת תרחישים (saved_scenarios)

| שדה | סוג | תיאור |
|-----|------|--------|
| id | INTEGER PRIMARY KEY | מזהה ייחודי |
| user_id | INTEGER FK | שיוך למשתמש |
| name | TEXT NOT NULL | שם התרחיש (ייחודי למשתמש) |
| payload_json | TEXT NOT NULL | גוף הבקשה בפורמט JSON |
| stop_count | INTEGER NOT NULL | מספר העצירות |
| created_at | TEXT NOT NULL | חותמת יצירה |
| updated_at | TEXT NOT NULL | חותמת עדכון אחרון |

#### 7.1.3 טבלת הרצות אופטימיזציה (optimization_runs)

| שדה | סוג | תיאור |
|-----|------|--------|
| id | INTEGER PRIMARY KEY | מזהה ייחודי |
| user_id | INTEGER FK | שיוך למשתמש |
| scenario_id | INTEGER FK (nullable) | קישור לתרחיש (אופציונלי) |
| scenario_name | TEXT (nullable) | שם התרחיש בעת ההרצה |
| request_json | TEXT NOT NULL | בקשת האופטימיזציה המלאה |
| response_json | TEXT NOT NULL | תגובת האופטימיזציה המלאה |
| algorithm_used | TEXT NOT NULL | שם האלגוריתם שהופעל |
| optimized_cost | REAL NOT NULL | העלות הסופית |
| improvement_percent | REAL NOT NULL | אחוז השיפור |
| feasible | INTEGER NOT NULL | האם המסלול ישים (0/1) |
| stop_count | INTEGER NOT NULL | מספר עצירות |
| created_at | TEXT NOT NULL | חותמת זמן ההרצה |

### 7.2 מודלי נתונים (Pydantic)

#### 7.2.1 מודל עצירה (Stop)

```python
class Stop(BaseModel):
    id: int                    # מזהה ייחודי
    x: float                   # קואורדינטת X
    y: float                   # קואורדינטת Y
    lat: Optional[float]       # קו רוחב (אופציונלי)
    lng: Optional[float]       # קו אורך (אופציונלי)
    label: Optional[str]       # תווית תצוגה
    city: Optional[str]        # עיר
    country: Optional[str]     # מדינה
    street: Optional[str]      # רחוב
    house_number: Optional[str]# מספר בית
    window_start: float        # תחילת חלון זמן
    window_end: float          # סיום חלון זמן
    service_time: float        # זמן שירות (≥ 0)
    priority: int              # עדיפות (1–5)
```

#### 7.2.2 מודל מחסן (Depot)

```python
class Depot(BaseModel):
    x: float
    y: float
    lat: Optional[float]
    lng: Optional[float]
    label: Optional[str]
    city: Optional[str]
    country: Optional[str]
    street: Optional[str]
    house_number: Optional[str]
```

#### 7.2.3 מודל משקלות (Weights)

```python
class Weights(BaseModel):
    w_dist: float = 1.0        # משקל מרחק
    w_wait: float = 1.0        # משקל המתנה
    w_late: float = 2.0        # משקל איחור
    w_priority: float = 2.0    # משקל עדיפות
    w_shift: float = 4.0       # משקל חריגת משמרת
```

#### 7.2.4 מודל תצורת אופטימיזציה (OptimizationConfig)

```python
class OptimizationConfig(BaseModel):
    algorithm: Literal["2opt", "swap", "relocate"] = "2opt"
    max_iterations: int = 1000          # מקסימום איטרציות
    no_improvement_limit: int = 100     # עצירה אחרי N איטרציות ללא שיפור
    time_limit: Optional[float] = None  # מגבלת זמן ריצה בשניות
```

#### 7.2.5 מודל בקשת אופטימיזציה (RouteRequest)

```python
class RouteRequest(BaseModel):
    depot: Depot
    stops: List[Stop]           # לפחות עצירה אחת, IDs ייחודיים
    max_shift_time: float       # > 0
    weights: Weights
    optimization: Optional[OptimizationConfig]
    travel_mode: Literal["driving", "walking"] = "driving"
```

#### 7.2.6 מודל תגובת אופטימיזציה (RouteResponse)

```python
class RouteResponse(BaseModel):
    initial_route_order: List[int]      # סדר ראשוני
    optimized_route_order: List[int]    # סדר אופטימלי
    initial_cost: float                 # עלות ראשונית
    optimized_cost: float               # עלות סופית
    total_travel_time: float            # זמן נסיעה כולל
    total_wait_time: float              # זמן המתנה כולל
    total_lateness: float               # איחור כולל
    priority_adjusted_lateness: float   # איחור משוקלל לפי עדיפות
    total_penalty: float                # קנס כולל
    distance_cost: float                # עלות מרחק
    wait_cost: float                    # עלות המתנה
    lateness_cost: float                # עלות איחור
    shift_overrun: float                # חריגת משמרת
    shift_overrun_cost: float           # עלות חריגת משמרת
    finish_time: float                  # זמן סיום
    feasible: bool                      # ישימות
    improvement_percent: float          # אחוז שיפור
    improvement_found: bool             # האם נמצא שיפור
    iterations_used: int                # מספר איטרציות
    stopped_by: str                     # סיבת עצירה
    algorithm_used: str                 # אלגוריתם שהורץ
    visits: List[VisitResult]           # פרטי ביקור לכל עצירה
```

### 7.3 ממשק ה-API

#### 7.3.1 נקודות קצה – אימות

| שיטה | נתיב | תיאור | אימות נדרש |
|-------|-------|--------|-------------|
| POST | /register | יצירת חשבון חדש | לא |
| POST | /login | קבלת טוקן JWT | לא |
| GET | /me | פרופיל המשתמש הנוכחי | כן |

#### 7.3.2 נקודות קצה – אופטימיזציה

| שיטה | נתיב | תיאור | אימות נדרש |
|-------|-------|--------|-------------|
| POST | /optimize | הרצת אופטימיזציה | כן |

פרמטר שאילתה אופציונלי: `?scenario_id=<id>` לקישור ההרצה לתרחיש.

#### 7.3.3 נקודות קצה – תרחישים

| שיטה | נתיב | תיאור | אימות נדרש |
|-------|-------|--------|-------------|
| GET | /scenarios | רשימת תרחישים שמורים | כן |
| POST | /scenarios | שמירה/עדכון תרחיש | כן |
| GET | /scenarios/{id} | קריאת תרחיש ספציפי | כן |
| DELETE | /scenarios/{id} | מחיקת תרחיש | כן |

#### 7.3.4 נקודות קצה – היסטוריה

| שיטה | נתיב | תיאור | אימות נדרש |
|-------|-------|--------|-------------|
| GET | /history | רשימת הרצות אחרונות (ברירת מחדל: 20, מקסימום: 100) | כן |
| GET | /history/{run_id} | פרטי הרצה ספציפית (בקשה + תגובה מלאה) | כן |

#### 7.3.5 נקודות קצה – ג'אוקודינג וניתוב

| שיטה | נתיב | תיאור | אימות נדרש |
|-------|-------|--------|-------------|
| GET | /geocode/search | חיפוש מיקום לפי שם (מנוע פנימי) | לא |
| POST | /route-road | חישוב מסלול כביש בין נקודות ציון | לא |

#### 7.3.6 נקודות קצה – כלליות

| שיטה | נתיב | תיאור | אימות נדרש |
|-------|-------|--------|-------------|
| GET | /health | בדיקת תקינות: `{"status":"operational","version":"3.0"}` | לא |
| GET | / | הגשת ממשק המשתמש (SPA) | לא |

### 7.4 תהליך אופטימיזציה

```
בקשה נכנסת (/optimize)
        │
        ▼
┌───────────────────────┐
│  בניית מטריצת מרחקים  │  ← בחירה אוטומטית: Euclidean / TravelTime(Haversine)
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  בנייה חמדנית (Greedy)│  ← מסלול ראשוני
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  הערכת מסלול ראשוני   │  ← initial_cost
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  חיפוש מקומי          │  ← 2-opt / Swap / Relocate
│  (איטרטיבי)           │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  חישוב שיפור          │  ← improvement_percent
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  שמירה בהיסטוריה      │  ← record_optimization_run
└───────────┬───────────┘
            │
            ▼
      תגובה (RouteResponse)
```

### 7.5 מטריצת מרחקים (Distance Matrix)

המערכת מחשבת מראש מטריצת מרחקים N×N בין כל הנקודות (כולל מחסן) כדי להימנע מחישובים חוזרים:

```python
@dataclass(frozen=True)
class DistanceMatrix:
    values: dict[str, dict[str, float]]

    def between(self, a, b) -> float:
        return self.values[node_key(a)][node_key(b)]
```

הבחירה בין ספקי מרחק נעשית אוטומטית:
- אם למחסן יש `lat` ו-`lng` → `TravelTimeProvider` (Haversine → דקות)
- אחרת → `EuclideanDistanceProvider` (חסר ממד)

---

## 8. מימוש

### 8.1 מודול אימות (auth.py)

#### רישום משתמש

1. ולידציה: שם משתמש (regex: `^[A-Za-z0-9_-]{3,32}$`), דוא"ל, סיסמה (≥8 תווים)
2. נורמליזציה: trim + lowercase לדוא"ל
3. בדיקת כפילויות (שם משתמש ודוא"ל)
4. הצפנת סיסמה: `hashlib.scrypt(N=16384, r=8, p=1, dklen=32)` עם salt אקראי של 16 בתים
5. שמירה ב-SQLite
6. הנפקת טוקן JWT

#### התחברות

1. חיפוש משתמש לפי identity (שם משתמש או דוא"ל)
2. אימות סיסמה: השוואה בטוחה עם `hmac.compare_digest`
3. הנפקת טוקן JWT

#### JWT

- אלגוריתם: HS256
- מפתח סודי: 48 בתים (נוצר אוטומטית בהפעלה ראשונה או מסופק דרך משתנה סביבה)
- תוקף: 7 ימים
- מבנה: `{"sub": "<user_id>", "exp": <timestamp>, "iat": <timestamp>}`
- מימוש עצמאי ללא ספריות חיצוניות (hmac + hashlib + base64)

### 8.2 מודול אופטימיזציה (optimizer.py)

#### בנייה חמדנית (greedy_initial_route)

```
function greedy_initial_route(depot, stops, max_shift_time, weights):
    unvisited ← copy(stops)
    route ← []
    current ← depot
    current_time ← 0

    while unvisited is not empty:
        best_feasible ← null
        best_fallback ← null

        for each stop in unvisited:
            travel ← distance(current, stop)
            arrival ← current_time + travel
            wait ← max(0, stop.window_start - arrival)
            lateness ← max(0, arrival - stop.window_end)
            score ← w_dist × travel + w_wait × wait + w_late × lateness × priority_factor(stop.priority)

            evaluate full route with this stop appended
            if feasible and score < best_feasible_score:
                best_feasible ← stop
            if candidate_cost < best_fallback_score:
                best_fallback ← stop

        chosen ← best_feasible or best_fallback
        route.append(chosen)
        update current_time
        remove chosen from unvisited

    return route
```

#### שיפור 2-opt (improve_route_2opt)

```
function improve_route_2opt(depot, route, max_shift_time, weights, config):
    best_route ← copy(route)
    best_eval ← evaluate(best_route)

    while not stopping_condition:
        improved ← false
        for i in range(len - 1):
            for j in range(i+1, len):
                candidate ← reverse segment [i..j]
                candidate_eval ← evaluate(candidate)
                if is_better(candidate_eval, best_eval):
                    best_route ← candidate
                    best_eval ← candidate_eval
                    improved ← true
                    break
            if improved: break

        if not improved:
            no_improvement_count += 1

    return best_route, best_eval, metadata
```

#### קריטריון השוואה (is_better_evaluation)

```python
def is_better_evaluation(candidate_eval, current_eval):
    if candidate_eval["feasible"] != current_eval["feasible"]:
        return candidate_eval["feasible"]  # ישים עדיף תמיד
    return candidate_eval["total_cost"] < current_eval["total_cost"]
```

### 8.3 מודול הערכה (evaluator.py)

מודול ההערכה מחשב את כל מדדי העלות למסלול נתון:

1. עבור כל עצירה בסדר הנתון:
   - חישוב זמן הגעה = זמן נוכחי + מרחק מהנקודה הקודמת
   - חישוב המתנה = max(0, window_start - arrival)
   - חישוב תחילת שירות = arrival + wait
   - חישוב איחור = max(0, start_service - window_end)
   - חישוב סיום = start_service + service_time
   - צבירת סטטיסטיקות

2. חישוב חזרה למחסן

3. חישוב חריגת משמרת = max(0, finish_time - max_shift_time)

4. חישוב עלויות:
   - distance_cost = w_dist × total_travel
   - wait_cost = w_wait × total_wait
   - lateness_cost = w_late × priority_adjusted_lateness
   - shift_overrun_cost = w_shift × shift_overrun
   - total_penalty = wait_cost + lateness_cost + shift_overrun_cost
   - total_cost = distance_cost + total_penalty

### 8.4 מודול מרחקים (utils.py)

#### ספק מרחק אוקלידי

```python
class EuclideanDistanceProvider:
    def distance(self, a, b) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)
```

#### ספק מרחק Haversine

מחשב מרחק Great-circle בקילומטרים:

```python
class HaversineDistanceProvider:
    def distance(self, a, b) -> float:
        # חישוב Haversine עם EARTH_RADIUS_KM = 6371.0
        ...
```

#### ספק זמן נסיעה

ממיר מרחק Haversine לזמן נסיעה בדקות:

```python
class TravelTimeProvider:
    SPEED_KMH = {"driving": 50.0, "walking": 5.0}

    def distance(self, a, b) -> float:
        km = haversine.distance(a, b)
        return km / speed_kmh * 60.0  # דקות
```

### 8.5 מודול הגבלת קצב (rate_limit.py)

מגביל קצב מבוסס חלון הזזה (Sliding Window) בזיכרון:

| נקודת קצה | מגבלה | חלון | מזהה |
|------------|--------|-------|-------|
| POST /register | 5 בקשות | 60 שניות | לפי IP |
| POST /login | 10 בקשות | 60 שניות | לפי IP |
| POST /optimize | 30 בקשות | 60 שניות | לפי משתמש |
| POST /scenarios | 20 בקשות | 60 שניות | לפי משתמש |

חריגה מהמגבלה מחזירה HTTP 429 עם כותרת `Retry-After`.

### 8.6 מודול אחסון (storage.py)

#### שמירת תרחיש

- אם קיים תרחיש באותו שם למשתמש → עדכון (UPDATE)
- אחרת → יצירה חדשה (INSERT)
- הנתונים נשמרים כ-JSON מלא של RouteRequest

#### שמירת הרצת אופטימיזציה

- שומר את הבקשה המלאה והתגובה המלאה כ-JSON
- מקשר לתרחיש (אם סופק scenario_id תקף)
- שומר מטא-דאטה: אלגוריתם, עלות, שיפור, ישימות, מספר עצירות

### 8.7 ג'אוקודר מקומי (main.py)

המערכת כוללת ג'אוקודר פנימי עם מאגר מיקומים מוכרים:
- King's Cross Station, London
- Tower Bridge, London
- Hyde Park Corner, London
- Canary Wharf, London
- Times Square, New York
- Empire State Building, New York
- Eiffel Tower, Paris

חיפוש מתבצע לפי התאמת שם חלקית, עם אפשרות לסינון לפי קוד מדינה.

### 8.8 ניתוב כביש מקומי (main.py – /route-road)

המערכת מספקת ניתוב כביש פנימי (ללא שירות חיצוני) המחשב:
- מרחק: Haversine × מקדם כביש (1.28 לנהיגה, 1.12 להליכה)
- זמן: מרחק / מהירות (34 קמ"ש לנהיגה, 4.8 קמ"ש להליכה)
- גיאומטריה: נקודות ביניים ליצירת עקומה ויזואלית על המפה

### 8.9 שרת ראשי (main.py)

#### Middleware

1. **CORS** – `allow_origins=["*"]`, `allow_credentials=False` – מאפשר גישה מכל מקור (אבטחה באמצעות Bearer tokens)
2. **Request Logging** – לוגים לכל בקשה: שיטה, נתיב, סטטוס, זמן תגובה (ms)
3. **Static Files** – הגשת קבצים סטטיים מתיקיית `static/`

#### Lifespan

בעת הפעלת האפליקציה:
1. `init_auth_storage()` – יצירת טבלת users
2. `init_app_storage()` – יצירת טבלאות scenarios ו-optimization_runs

---

## 9. ממשק משתמש

### 9.1 סקירה כללית

ממשק המשתמש הוא אפליקציית דף יחיד (SPA) הכתובה ב-HTML, CSS ו-JavaScript טהור (Vanilla). הממשק מוגש ישירות מהשרת בנתיב `/` ואינו דורש שרת נפרד או כלי בנייה.

### 9.2 רכיבים עיקריים

#### 9.2.1 מערכת אימות

- מודל (Modal) לרישום והתחברות
- תמיכה ברישום (שם, דוא"ל, סיסמה) ובהתחברות (identity + סיסמה)
- שמירת טוקן ב-localStorage
- הצגת שם המשתמש המחובר בכותרת
- כפתור התנתקות

#### 9.2.2 בורר מדינה (Country Picker)

- רשת מדינות ויזואלית עם דגלים
- סינון לפי שם
- השפעה על ג'אוקודינג (מגביל חיפוש למדינה הנבחרת)

#### 9.2.3 הגדרת מחסן

- שדות כתובת: רחוב, מספר בית, עיר
- ג'אוקודינג אוטומטי (חיפוש בהקלדה עם השלמה אוטומטית)
- שדות קואורדינטות מספריות (lat, lng, x, y)
- אינדיקטור סטטוס (✓ נפתר / ⚠ לא נפתר)

#### 9.2.4 הגדרת עצירות

- כרטיסיות עצירה דינמיות (הוספה/הסרה)
- לכל עצירה: כתובת, קואורדינטות, חלון זמן (start/end), זמן שירות, עדיפות
- ג'אוקודינג אוטומטי לכל עצירה
- אינדיקטור סטטוס לכל עצירה

#### 9.2.5 הגדרות אופטימיזציה

- בחירת אלגוריתם (2-opt / Swap / Relocate)
- מקסימום איטרציות
- מגבלת איטרציות ללא שיפור
- מגבלת זמן ריצה
- זמן משמרת מקסימלי
- מצב נסיעה (נהיגה / הליכה)

#### 9.2.6 משקלות

- מחוונים (sliders) או שדות מספריים לכל משקל:
  - משקל מרחק
  - משקל המתנה
  - משקל איחור
  - משקל עדיפות
  - משקל חריגת משמרת

#### 9.2.7 תוצאות

- כרטיסיות מדדים: עלות ראשונית, עלות סופית, אחוז שיפור, זמן נסיעה, ישימות
- טבלת ביקורים: לכל עצירה – הגעה, תחילת שירות, סיום, המתנה, איחור
- מטא-דאטה: אלגוריתם, איטרציות, סיבת עצירה

#### 9.2.8 ויזואליזציה – מפה

- מפה אינטראקטיבית מבוססת Leaflet.js
- סגנונות מפה: Streets, Satellite, Topo, Dark
- סמנים (Markers): מחסן (ירוק), עצירות (ממוספרות בסדר אופטימלי)
- קווי מסלול: ניתוב כביש מציאותי (לא קו ישר)
- פאנל הוראות ניווט (Turn-by-turn)
- סטטיסטיקות מסלול: מרחק כולל, זמן כולל

#### 9.2.9 ניהול תרחישים

- שמירת תרחיש בשם
- טעינת תרחיש שמור (מילוי אוטומטי של כל השדות)
- מחיקת תרחיש
- רשימת תרחישים שמורים

#### 9.2.10 היסטוריית אופטימיזציות

- רשימת הרצות אחרונות
- לכל הרצה: תאריך, אלגוריתם, עלות, שיפור, ישימות, מספר עצירות
- צפייה בפרטי הרצה מלאים (בקשה + תגובה)

### 9.3 מצב כהה (Dark Mode)

הממשק תומך במעבר חלק בין מצב בהיר למצב כהה:
- כפתור מעבר בכותרת
- שמירת ההעדפה ב-localStorage
- אנימציית מעבר חלקה (280ms cubic-bezier)
- התאמה אוטומטית של כל הרכיבים כולל המפה

### 9.4 רספונסיביות

הממשק מותאם למסכים שונים באמצעות:
- CSS Grid ו-Flexbox
- Media queries
- יחידות יחסיות (rem, %, vw)
- מקסימום רוחב 1440px עם padding דינמי

---

## 10. בדיקות

### 10.1 סקירה כללית

המערכת כוללת חבילת בדיקות מקיפה הכתובה ב-Python עם pytest ו-httpx (TestClient).

### 10.2 הרצת הבדיקות

```bash
cd routemind
pip install -r requirements.txt
pip install pytest httpx
python -m pytest tests/test_routemind.py -v
```

### 10.3 סביבת בדיקות

- ספריית אימות זמנית: `tests/.test_auth_store`
- משתני סביבה: `ROUTEMIND_AUTH_DIR`, `ROUTEMIND_SECRET_KEY`
- בסיס נתונים נפרד לבדיקות (מבודד מסביבת ייצור)

### 10.4 קטגוריות בדיקות

#### 10.4.1 בדיקות יחידה – מנוע אופטימיזציה

| בדיקה | תיאור |
|--------|--------|
| test_distance_calculation | חישוב מרחק אוקלידי בין שתי נקודות |
| test_priority_factor | חישוב מקדם עדיפות לערכי priority שונים |
| test_evaluate_single_stop | הערכת מסלול עם עצירה אחת |
| test_evaluate_multiple_stops | הערכת מסלול עם מספר עצירות |
| test_evaluate_shift_overrun | זיהוי חריגת משמרת |
| test_greedy_route_construction | בדיקת בנייה חמדנית |
| test_greedy_feasibility_preference | העדפת מסלול ישים על פני זול יותר |
| test_2opt_improvement | שיפור מסלול עם 2-opt |
| test_swap_improvement | שיפור מסלול עם Swap |
| test_relocate_improvement | שיפור מסלול עם Relocate |
| test_stopping_criteria | תנאי עצירה: max_iterations, no_improvement, time_limit |
| test_single_stop_route | מסלול עם עצירה בודדת (מקרה קצה) |

#### 10.4.2 בדיקות יחידה – מודלי Pydantic

| בדיקה | תיאור |
|--------|--------|
| test_stop_validation | ולידציית שדות עצירה |
| test_window_end_validation | window_end ≥ window_start |
| test_service_time_validation | service_time ≥ 0 |
| test_priority_range | priority ∈ [1,5] |
| test_unique_stop_ids | בדיקת ייחודיות מזהי עצירות |
| test_max_shift_time_positive | max_shift_time > 0 |
| test_weights_defaults | ערכי ברירת מחדל למשקלות |

#### 10.4.3 בדיקות יחידה – אימות

| בדיקה | תיאור |
|--------|--------|
| test_password_hashing | הצפנה ואימות סיסמה |
| test_jwt_creation_and_decode | יצירה ופענוח טוקן |
| test_jwt_expiry | טוקן פג תוקף |
| test_username_validation | ולידציית שם משתמש |
| test_email_validation | ולידציית דוא"ל |
| test_email_normalization | נורמליזציית דוא"ל |

#### 10.4.4 בדיקות אינטגרציה – HTTP API

| בדיקה | תיאור |
|--------|--------|
| test_health_endpoint | GET /health מחזיר status=operational |
| test_register_flow | רישום מוצלח |
| test_register_duplicate_username | רישום עם שם כפול → 400 |
| test_register_duplicate_email | רישום עם דוא"ל כפול → 400 |
| test_login_success | התחברות מוצלחת |
| test_login_invalid_password | סיסמה שגויה → 401 |
| test_optimize_unauthorized | אופטימיזציה ללא טוקן → 401 |
| test_optimize_success | הרצת אופטימיזציה מוצלחת |
| test_optimize_all_algorithms | בדיקת כל שלושת האלגוריתמים |
| test_scenarios_crud | יצירה, קריאה, עדכון, מחיקה של תרחישים |
| test_history_endpoints | בדיקת נקודות קצה של היסטוריה |
| test_rate_limiting | חריגה ממגבלת קצב → 429 |

#### 10.4.5 בדיקות מרחק גיאוגרפי

| בדיקה | תיאור |
|--------|--------|
| test_haversine_known_distance | חישוב Haversine למרחק ידוע |
| test_travel_time_provider | המרת מרחק לזמן נסיעה |
| test_distance_matrix_build | בניית מטריצת מרחקים |
| test_auto_provider_selection | בחירה אוטומטית של ספק מרחק |

### 10.5 כיסוי בדיקות

חבילת הבדיקות מכסה:
- לוגיקת אופטימיזציה (Greedy, 2-opt, Swap, Relocate)
- הערכת עלות מסלול
- חישובי מרחק (Euclidean, Haversine, TravelTime)
- ולידציית קלט (Pydantic models)
- מערכת אימות (רישום, התחברות, JWT)
- נקודות קצה HTTP (אינטגרציה מלאה)
- הגבלת קצב
- ניהול תרחישים
- היסטוריית אופטימיזציות

---

## 11. אבטחת מידע

### 11.1 הצפנת סיסמאות

הסיסמאות מוצפנות באמצעות `hashlib.scrypt` עם הפרמטרים הבאים:
- N = 16,384 (cost factor)
- r = 8 (block size)
- p = 1 (parallelization)
- dklen = 32 bytes (key length)
- salt = 16 bytes (random, per-password)

פורמט אחסון: `scrypt$N$r$p$<base64_salt>$<base64_hash>`

### 11.2 אימות JWT

- אלגוריתם: HS256 (HMAC-SHA256)
- מפתח סודי: 48 בתים שנוצרים אוטומטית (secrets.token_urlsafe)
- אחסון מפתח: קובץ `.secret_key` או משתנה סביבה `ROUTEMIND_SECRET_KEY`
- תוקף: 7 ימים (ACCESS_TOKEN_EXPIRE_MINUTES = 10,080)
- אימות חתימה: `hmac.compare_digest` (constant-time comparison)

### 11.3 CORS

- מקורות מותרים: `["*"]` (כל מקור)
- credentials: `False` – אין העברת cookies
- האבטחה מבוססת על Bearer tokens בכותרת Authorization

### 11.4 ולידציית קלט

- כל הקלטים עוברים ולידציה דרך מודלי Pydantic
- בדיקות טווח: priority ∈ [1,5], service_time ≥ 0, max_shift_time > 0
- בדיקות לוגיות: window_end ≥ window_start, stop IDs ייחודיים
- אורכי מחרוזות: username (3–32), password (8–128), scenario name (1–80)

### 11.5 הגבלת קצב

- מנגנון Sliding Window בזיכרון
- מגבלות ספציפיות לכל נקודת קצה
- זיהוי לפי IP (אנונימי) או user ID (מאומת)
- תגובה: HTTP 429 + Retry-After header

### 11.6 בסיס נתונים

- SQLite עם WAL mode (Write-Ahead Logging) לביצועים ויציבות
- PRAGMA synchronous=NORMAL
- Foreign keys מוגדרים אך SQLite לא אוכף אותם כברירת מחדל
- מניעת SQL injection: שימוש בפרמטרים (parameterized queries) בכל השאילתות

---

## 12. פריסה והרצה

### 12.1 הרצה מקומית

```bash
# התקנת תלויות
cd routemind
pip install -r requirements.txt

# הרצת השרת
uvicorn app.main:app --reload

# השרת זמין בכתובת http://localhost:8000
# ממשק משתמש: http://localhost:8000/
# תיעוד API: http://localhost:8000/docs
```

### 12.2 Docker

#### Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY routemind/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY routemind/ .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### בנייה והרצה

```bash
docker build -t routemind .
docker run -p 8000:8000 routemind
```

#### Docker Compose

```bash
docker compose up
```

### 12.3 משתני סביבה

| משתנה | תיאור | ברירת מחדל |
|--------|--------|-------------|
| ROUTEMIND_AUTH_DIR | ספריית בסיס נתונים האימות | `routemind/auth_store/` |
| ROUTEMIND_DB_PATH | נתיב מלא לקובץ בסיס הנתונים | `<AUTH_DIR>/routemind.db` |
| ROUTEMIND_SECRET_KEY_PATH | נתיב למפתח הסודי | `<AUTH_DIR>/.secret_key` |
| ROUTEMIND_SECRET_KEY | ערך מפתח סודי (עוקף קובץ) | נוצר אוטומטית |

### 12.4 תלויות

קובץ `requirements.txt`:
```
fastapi
uvicorn
pydantic
```

תלויות נוספות לבדיקות:
```
pytest
httpx
```

### 12.5 דרישות מערכת

- Python 3.11 ומעלה
- ללא צורך בבסיס נתונים חיצוני (SQLite מובנה)
- ללא צורך בשירותי צד שלישי לליבת האופטימיזציה

---

## 13. סיכום ומסקנות

### 13.1 סיכום הפרויקט

פרויקט RouteMind מממש מערכת אופטימיזציה למסלולי שילוח מלאה, הכוללת:

1. **מנוע אופטימיזציה** – בנייה חמדנית + שלושה אלגוריתמי חיפוש מקומי (2-opt, Swap, Relocate) עם תנאי עצירה גמישים
2. **מודל עלות מתוחכם** – פונקציית עלות רב-ממדית עם משקלות ניתנים לכיוונון, מערכת עדיפויות, ואכיפת מגבלות משמרת
3. **שרת API מאובטח** – FastAPI עם אימות JWT, הגבלת קצב, וולידציית קלט מלאה
4. **ממשק משתמש עשיר** – SPA אינטראקטיבי עם מפות, ג'אוקודינג, ויזואליזציה, מצב כהה, וניהול תרחישים
5. **תמיכה גיאוגרפית** – חישוב מרחקים Haversine, ניתוב כביש, והצגה על מפת Leaflet
6. **אחסון ותיעוד** – SQLite לניהול משתמשים, תרחישים, והיסטוריית הרצות
7. **פריסה** – תמיכה מלאה ב-Docker לפריסה מהירה

### 13.2 עמידה ביעדים

| יעד | מצב |
|-----|------|
| מנוע אופטימיזציה עם מספר אלגוריתמים | ✓ הושלם |
| שרת API מאובטח עם JWT | ✓ הושלם |
| ממשק דפדפן אינטראקטיבי עם מפות | ✓ הושלם |
| ניהול תרחישים והיסטוריה | ✓ הושלם |
| ג'אוקודינג ומרחקים גיאוגרפיים | ✓ הושלם |
| פריסה עם Docker | ✓ הושלם |

### 13.3 חידושים ותרומות

1. **פונקציית עלות גמישה** – שילוב ייחודי של מרחק, המתנה, איחור משוקלל-עדיפות, וחריגת משמרת
2. **מקדם עדיפות** – מנגנון שמאפשר לתעדף עצירות קריטיות מבלי לפגוע באופטימיזציה הכוללת
3. **בחירת ספק מרחק אוטומטית** – המערכת מזהה אוטומטית אם הנתונים גיאוגרפיים ובוחרת את שיטת החישוב המתאימה
4. **ארכיטקטורה עצמאית** – ללא תלות בשירותים חיצוניים, ניתנת לפריסה כ-container בודד

### 13.4 מגבלות ידועות

1. **רכב יחיד** – המערכת מטפלת במסלול של רכב בודד; אינה תומכת בפיצול לצי רכבים
2. **ג'אוקודר מוגבל** – הג'אוקודר הפנימי מכיל מספר מצומצם של מיקומים מוכרים; בייצור מומלץ לחבר שירות חיצוני
3. **ניתוב משוער** – ניתוב הכביש הפנימי הוא אומדן (מרחק × מקדם) ולא מבוסס רשת כבישים אמיתית
4. **אין אופטימיזציה גלובלית** – האלגוריתמים הם חיפוש מקומי ואינם מבטיחים אופטימום גלובלי
5. **SQLite** – מתאים לשימוש יחיד/קטן; לעומסים גבוהים מומלץ לעבור ל-PostgreSQL

### 13.5 כיווני פיתוח עתידיים

1. **תמיכה בצי רכבים** – הרחבה ל-Multi-Vehicle VRP
2. **אלגוריתמים מתקדמים** – הוספת Simulated Annealing, Genetic Algorithm, או Ant Colony Optimization
3. **אופטימיזציה בזמן אמת** – עדכון מסלול דינמי בהתאם לתנועה ושינויים
4. **אינטגרציה עם שירותי מפות** – Google Maps API, Mapbox, OSRM לניתוב מדויק
5. **אפליקציית מובייל** – נהג עם ניווט בזמן אמת
6. **דוחות וניתוח** – דשבורד עם סטטיסטיקות היסטוריות ומגמות
7. **ייצוא** – ייצוא מסלולים לפורמטים שונים (GPX, CSV, PDF)
8. **מערכת הרשאות** – תפקידים (admin, dispatcher, driver) עם הרשאות שונות

---

## 14. נספחים

### נספח א' – דוגמת בקשת אופטימיזציה

```json
{
  "depot": {
    "x": 0.0,
    "y": 0.0,
    "lat": 51.5308,
    "lng": -0.1238,
    "label": "King's Cross Station"
  },
  "stops": [
    {
      "id": 1,
      "x": 3.0,
      "y": 4.0,
      "lat": 51.5055,
      "lng": -0.0754,
      "label": "Tower Bridge",
      "window_start": 10.0,
      "window_end": 30.0,
      "service_time": 5.0,
      "priority": 3
    },
    {
      "id": 2,
      "x": 6.0,
      "y": 1.0,
      "lat": 51.5027,
      "lng": -0.1528,
      "label": "Hyde Park Corner",
      "window_start": 15.0,
      "window_end": 45.0,
      "service_time": 3.0,
      "priority": 5
    },
    {
      "id": 3,
      "x": 2.0,
      "y": 7.0,
      "lat": 51.5054,
      "lng": -0.0235,
      "label": "Canary Wharf",
      "window_start": 20.0,
      "window_end": 60.0,
      "service_time": 4.0,
      "priority": 1
    }
  ],
  "max_shift_time": 120.0,
  "weights": {
    "w_dist": 1.0,
    "w_wait": 1.0,
    "w_late": 2.0,
    "w_priority": 2.0,
    "w_shift": 4.0
  },
  "optimization": {
    "algorithm": "2opt",
    "max_iterations": 1000,
    "no_improvement_limit": 100,
    "time_limit": 10.0
  },
  "travel_mode": "driving"
}
```

### נספח ב' – דוגמת תגובת אופטימיזציה

```json
{
  "initial_route_order": [1, 2, 3],
  "optimized_route_order": [2, 1, 3],
  "initial_cost": 45.7,
  "optimized_cost": 38.2,
  "total_travel_time": 22.5,
  "total_wait_time": 3.1,
  "total_lateness": 0.0,
  "priority_adjusted_lateness": 0.0,
  "total_penalty": 3.1,
  "distance_cost": 22.5,
  "wait_cost": 3.1,
  "lateness_cost": 0.0,
  "shift_overrun": 0.0,
  "shift_overrun_cost": 0.0,
  "finish_time": 52.6,
  "feasible": true,
  "improvement_percent": 16.4,
  "improvement_found": true,
  "iterations_used": 47,
  "stopped_by": "no_improvement_limit",
  "algorithm_used": "2opt",
  "visits": [
    {
      "stop_id": 2,
      "arrival": 8.3,
      "start_service": 15.0,
      "finish": 18.0,
      "wait": 6.7,
      "lateness": 0.0
    },
    {
      "stop_id": 1,
      "arrival": 22.1,
      "start_service": 22.1,
      "finish": 27.1,
      "wait": 0.0,
      "lateness": 0.0
    },
    {
      "stop_id": 3,
      "arrival": 35.4,
      "start_service": 35.4,
      "finish": 39.4,
      "wait": 0.0,
      "lateness": 0.0
    }
  ]
}
```

### נספח ג' – דוגמת רישום והתחברות

#### בקשת רישום

```http
POST /register
Content-Type: application/json

{
  "username": "driver_1",
  "email": "driver1@company.com",
  "password": "secureP@ss123"
}
```

#### תגובת רישום

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "id": 1,
    "username": "driver_1",
    "email": "driver1@company.com",
    "created_at": "2026-01-15T10:30:00+00:00"
  }
}
```

#### בקשת התחברות

```http
POST /login
Content-Type: application/json

{
  "identity": "driver_1",
  "password": "secureP@ss123"
}
```

### נספח ד' – ניהול תרחישים

#### שמירת תרחיש

```http
POST /scenarios
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "London Morning Route",
  "route": { ... }
}
```

#### תגובה

```json
{
  "id": 1,
  "name": "London Morning Route",
  "stop_count": 3,
  "created_at": "2026-01-15T10:35:00+00:00",
  "updated_at": "2026-01-15T10:35:00+00:00",
  "route": { ... }
}
```

### נספח ה' – מילון מונחים

| מונח | הסבר |
|------|--------|
| VRP | Vehicle Routing Problem – בעיית ניתוב רכבים |
| VRPTW | Vehicle Routing Problem with Time Windows – VRP עם חלונות זמן |
| Depot | מחסן – נקודת המוצא והסיום של המסלול |
| Stop | עצירה – נקודה במסלול שיש לבקר בה |
| Time Window | חלון זמן – טווח הזמנים המותר לתחילת שירות |
| Service Time | זמן שירות – משך הזמן הנדרש בעצירה |
| Lateness | איחור – הגעה אחרי סוף חלון הזמן |
| Shift Overrun | חריגת משמרת – סיום מסלול אחרי הזמן המקסימלי |
| Feasible | ישים – מסלול שאינו חורג מזמן המשמרת |
| 2-opt | אלגוריתם חיפוש מקומי המבוסס על היפוך תת-מקטע |
| Swap | החלפה – אלגוריתם המחליף בין זוגות עצירות |
| Relocate | העברה – אלגוריתם המזיז עצירה למיקום אחר במסלול |
| Greedy | חמדני – אלגוריתם הבוחר בכל צעד את האופציה הטובה ביותר מקומית |
| Haversine | נוסחה לחישוב מרחק Great-circle על פני כדור הארץ |
| JWT | JSON Web Token – טוקן אימות מבוסס JSON |
| scrypt | אלגוריתם הצפנת סיסמאות עמיד בפני brute-force |
| Rate Limiting | הגבלת קצב – מנגנון למניעת שימוש לרעה |
| SPA | Single Page Application – אפליקציית דף יחיד |
| CORS | Cross-Origin Resource Sharing – שיתוף משאבים בין מקורות |
| SQLite | מנוע בסיס נתונים קל המוטמע באפליקציה |
| FastAPI | מסגרת Python מודרנית לבניית APIs |
| Pydantic | ספריית ולידציה וסריאליזציה ל-Python |
| Uvicorn | שרת ASGI מבוסס uvloop |
| Leaflet.js | ספריית JavaScript למפות אינטראקטיביות |

### נספח ו' – רשימת קבצי קוד מקור

| קובץ | שורות (בקירוב) | תפקיד |
|-------|----------------|--------|
| app/main.py | 516 | אפליקציה ראשית, נתיבי API, middleware |
| app/models.py | 214 | מודלי נתונים Pydantic |
| app/auth.py | 305 | מערכת אימות, JWT, הצפנת סיסמאות |
| app/optimizer.py | 323 | אלגוריתמי אופטימיזציה |
| app/evaluator.py | 79 | הערכת עלות מסלול |
| app/utils.py | 124 | חישובי מרחק, מטריצת מרחקים |
| app/rate_limit.py | 31 | הגבלת קצב |
| app/storage.py | 258 | אחסון תרחישים והיסטוריה |
| static/index.html | ~4600 | ממשק משתמש מלא (HTML+CSS+JS) |
| tests/test_routemind.py | ~560 | חבילת בדיקות |

---

*סוף המסמך*

</div>
