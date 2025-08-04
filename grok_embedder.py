import re
from typing import Dict, List, Optional

class GrokEmbedder:
    def __init__(self):
        self.text_chunks = []  # list of {"page": int, "chunk": str}
        self.entities = {}

    def create_index(self, text_dict: Dict[int, str]):
        """
        Breaks PDF text per page into chunks and extracts structured entities.
        """
        all_text = " ".join(text_dict.values())
        self.entities = self.extract_entities(all_text)

        self.text_chunks = []
        for page_num, text in text_dict.items():
            paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 30]
            for para in paragraphs:
                self.text_chunks.append({"page": page_num, "chunk": para})

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Use simple regex to extract structured data.
        """
        return {
            "emails": re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text),
            "phones": re.findall(r'\+?\d[\d\s\-\(\)]{8,}\d', text),
            "names": re.findall(r'(?i)(?:resume|cv)?\s*([A-Z][a-z]+ [A-Z][a-z]+)', text[:200]),
            "education": re.findall(r'(?i)(Bachelor|Master|B\.Tech|M\.Tech|BSc|MSc|PhD)[^,\n]+', text),
            "experience": re.findall(r'(?i)(\d+ [+-]?\s?(years|yrs) of experience.*?)\.', text),
            "skills": re.findall(r'(?i)(?:(skills|technologies)[:\-]?\s*)([\w\s,\/\+]+)', text)
        }

    def analyze_and_answer(self, question: str) -> Dict[str, Optional[str]]:
        """
        Main interface to respond to a question.
        """
        question_lower = question.lower()
        intent = self._infer_intent(question_lower)

        if intent == "name" and self.entities['names']:
            return self._make_response(f"Name: {self.entities['names'][0]}", 1)

        if intent == "email" and self.entities['emails']:
            return self._make_response(f"Email: {self.entities['emails'][0]}", 1)

        if intent == "phone" and self.entities['phones']:
            return self._make_response(f"Phone: {self.entities['phones'][0]}", 1)

        if intent == "education" and self.entities['education']:
            educ = ", ".join(set(self.entities['education']))
            return self._make_response("Education: " + educ, self._find_page("education"))

        if intent == "experience" and self.entities['experience']:
            exp = ", ".join([e[0] for e in self.entities['experience']])
            return self._make_response("Experience: " + exp, self._find_page("experience"))

        if intent == "skills" and self.entities['skills']:
            skill_strings = [s[1] for s in self.entities['skills']]
            skill_set = ", ".join(set(", ".join(skill_strings).split(',')))
            return self._make_response("Skills: " + skill_set, self._find_page("skills"))

        # fallback: search best-matching chunk
        match = self._search_text_chunks(question_lower)
        if match:
            return self._make_response(f"Relevant content: {match['chunk']}", match['page'])

        return self._make_response("Sorry, I couldn't find relevant information.", None)

    def _infer_intent(self, question: str) -> str:
        """
        Determine the type of question asked.
        """
        intent_keywords = {
            "name": ["name", "who is", "called"],
            "email": ["email", "contact"],
            "phone": ["phone", "mobile", "number"],
            "education": ["education", "degree", "university", "college", "graduated"],
            "experience": ["experience", "years", "worked", "career", "job"],
            "skills": ["skills", "stack", "technologies", "tools", "languages", "framework"]
        }
        for intent, keywords in intent_keywords.items():
            if any(kw in question for kw in keywords):
                return intent
        return "general"

    def _search_text_chunks(self, question: str) -> Optional[Dict]:
        """
        Fallback search: find text chunk with max keyword overlap.
        """
        best = None
        max_overlap = 0
        q_words = set(question.split())

        for chunk in self.text_chunks:
            chunk_words = set(chunk["chunk"].lower().split())
            overlap = len(q_words.intersection(chunk_words))
            if overlap > max_overlap:
                best = chunk
                max_overlap = overlap

        return best

    def _find_page(self, keyword: str) -> int:
        """
        Get the first page number that contains the keyword.
        """
        for chunk in self.text_chunks:
            if keyword.lower() in chunk['chunk'].lower():
                return chunk['page']
        return 1

    def _make_response(self, answer: str, page_number: Optional[int]) -> Dict[str, Optional[str]]:
        """
        Final response format with citation.
        """
        return {
            "answer": answer,
            "citation": f"Page {page_number}" if page_number else None
        }
