# Wykorzystanie pamięci: Mechanizmy pobierania i optymalizacja sesji

Niniejszy dokument opisuje strategię integracji trwałej bazy wiedzy z kontekstem agenta LLM, kładąc nacisk na niskie opóźnienia, efektywność kosztową i wysoką precyzję faktograficzną.

---

## 1. Strategie dostarczania kontekstu

### Podejście A: Pełne wstrzyknięcie profilu
Wstawianie całego profilu użytkownika bezpośrednio do promptu systemowego.

* **Wydajność**: Najniższe opóźnienia (brak dodatkowych zapytań).
* **Niezawodność**: Najwyższa (pełna widoczność danych dla LLM).
* **Skalowanie**: Ograniczone do małych profilów (< 20-30 faktów).

### Podejście B: Adaptacyjny RAG (Retrieval-Augmented Generation)
Dynamiczne pobieranie najbardziej istotnych faktów na podstawie bieżącej tury rozmowy.

* **Efektywność**: Zoptymalizowany koszt.
* **Opóźnienia**: Średnie (+50-150ms na wyszukiwanie wektorowe).
* **Implementacja**: Indeksowanie faktów w bazie wektorowej (np. ElasticSearch z polem `dense_vector`).

### Podejście C: Profil narracyjny
Przygotowanie przez LLM spójnego, opisowego podsumowania użytkownika.

* **Zastosowanie**: Companion AI, gdzie liczy się naturalność.
* **Ryzyko**: Wysoki potencjał halucynacji.

---

## 2. Rekomendowana architektura hybrydowa

Dla systemów produkcyjnych (E-commerce, Wsparcie klienta) stosowane jest podejście hybrydowe:

1. **Kontekst podstawowy**: Kluczowe cechy (język, lokalizacja, główne filtry) są zawsze obecne w prompcie.
2. **Pobieranie zależne od rozmiaru**:
    - Profile < 20 faktów: Pełne wstrzyknięcie.
    - Profile > 20 faktów: Mechanizm RAG.
3. **Zunifikowane repozytorium**: Wykorzystanie **ElasticSearch** jako silnika dla zapytań klucz-wartość oraz wyszukiwania wektorowego.

---

## 3. Dynamiczna pamięć w dialogach wieloturowych

Pamięć jest procesem reaktywnym, zmieniającym się wraz z tematem rozmowy:

1. **Wektoryzacja tury**: Każda wiadomość jest wektoryzowana w tle.
2. **Logika wykluczania**: System śledzi `already_injected_fact_ids`, aby uniknąć dublowania kontekstu w tej samej sesji.
3. **Przypomnienia systemowe**: Nowe fakty są dołączane jako ukryte przypomnienia przed najnowszą wiadomością użytkownika.

### Mechanizmy kontroli kosztów

* **Filtrowanie wiadomości**: Pomijanie wyszukiwania dla krótkich wypowiedzi (np. "Ok", "Tak").
* **Limity wstrzyknięć**: Maksymalna liczba aktualizacji kontekstu na sesję (np. limit 10 nowych faktów).
