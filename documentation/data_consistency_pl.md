# Spójność danych: Deduplikacja i rozwiązywanie konfliktów

W miarę rozwoju bazy wiedzy na przestrzeni wielu sesji, utrzymanie wysokiej jakości danych wymaga rozwiązania dwóch kluczowych wyzwań:
* **Nadmiarowość** — identyfikacja tego samego faktu opisanego różnymi słowami.
* **Dezaktualizacja wiedzy** — zarządzanie informacjami, które zostały zastąpione przez nowsze wydarzenia.

Poniższa architektura zapewnia produkcyjny mechanizm zarządzania tymi wyzwaniami.

---

## 1. Deduplikacja faktów

Aby uniknąć nadmiarowości w przechowywaniu i przetwarzaniu (np. "Mam psa" vs "Moim zwierzakiem jest pies"), system wykorzystuje wieloetapową strategię deduplikacji.

### Podobieństwo semantyczne (Embeddings)

1. **Grupowanie po kategoriach**: Porównania są ograniczone do faktów w obrębie tej samej kategorii (np. `household`), aby zminimalizować narzut obliczeniowy.
2. **Wektoryzacja**: Nowe fakty są zamieniane na wysokowymiarowe embeddingi (np. `text-embedding-004`).
3. **Ocena podobieństwa**:
   - **Wysoka pewność (≥ 0.92)**: Automatycznie traktowane jako duplikat. System aktualizuje znacznik czasu `last_confirmed_at` i zwiększa licznik `confirmation_count` w istniejącym wpisie.
   - **Strefa niepewności (0.89 – 0.92)**: Uruchamia krok weryfikacji przez model LLM z precyzyjnym promptem: *"Czy zdanie A i zdanie B reprezentują identyczną cechę użytkownika?"*

---

## 2. Rozwiązywanie konfliktów

Preferencje użytkowników i okoliczności życiowe są dynamiczne. System musi rozpoznawać, kiedy nowe informacje unieważniają historyczne rekordy.

### Mapowanie kluczy faktów

Fakty zidentyfikowane jako "unikalne w danym momencie" (np. `marital_status`, `working_location`) otrzymują **Klucz Funkcjonalny (`fact_key`)**. Gdy pojawia się nowy fakt dla istniejącego klucza:

1. **Autorytet chronologiczny**: Najnowsza ekstrakcja jest traktowana jako stan "Aktywny".
2. **Miękkie usuwanie i ciągłość**: Historyczne rekordy pozostają w bazie danych, ale otrzymują flagę `is_outdated = True`.
3. **Łańcuch zastępowania**: Dezaktualizowane rekordy zawierają wskaźnik `superseded_by` do nowego ID rekordu.

---

## 3. Na co uważać?

* **Logika wewnątrz wiadomości**: Pipeline ekstrakcji musi priorytetyzować ostateczną intencję użytkownika (np. "Szukałem auta... a jednak się wstrzymam").
* **Ubytek zaufania (Confidence decay)**: Fakty, które nie zostały potwierdzone przez długi czas, powinny mieć systemowo obniżany wynik pewności.
* **Hipotetyczność i plany**: Ścisłe filtrowanie wypowiedzi przypuszczających ("Chciałbym mieć...") odbywa się na poziomie inżynierii promptów.

---

## 4. Rozszerzony schemat metadanych

| Pole | Opis |
|---|---|
| `fact_key` | Identyfikator wykluczających się cech. |
| `confidence` | Wynik pewności (0.0 - 1.0) przypisany przez model. |
| `is_outdated` | Flaga logiczna dla zastąpionej wiedzy. |
| `source_conversation_id` | Link do źródłowego pliku rozmowy. |
